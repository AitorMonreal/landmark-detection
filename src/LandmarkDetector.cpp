#include "LandmarkDetector.h"


cam::LandmarkDetector::LandmarkDetector() :
    StreamConsumer(),
    mFrame(nullptr),
    mExit(false)
{
    try {
        mModel = cv::dnn::readNetFromONNX(mModelFilePath);
    } catch(cv::Exception & e) {
        Logger::suicide(std::format("read model exception: {}", e.what()));
    }
    if (mModel.empty()) {
        Logger::suicide("failed to load the ONNX model");
    }

    // start thread
    mThread = std::thread(&LandmarkDetector::run, this);
}


cam::LandmarkDetector::~LandmarkDetector() {

    // signal thread
    std::unique_lock<std::mutex> lock(mLock);

    mExit = true;

    // clean any remaining promises + set their success & message
    TLandmarkDetectorTransaction returnValue = TLandmarkDetectorTransaction();
    returnValue.mSuccess = false;
    returnValue.mMessage = "exiting thread, promises not processed";
    for(auto & promise : mPromises) {
        promise.set_value(returnValue);
    }
    mPromises.clear();

    lock.unlock();
    mConditionVariable.notify_all();

    mThread.join();  // Wait for the thread to finish
}




std::future<cam::TLandmarkDetectorTransaction> cam::LandmarkDetector::predict() {
    std::unique_lock<std::mutex> lock(mLock);

    if(mExit) {
        Logger::error("exit flag is set");
        return;
    }

    std::promise<TLandmarkDetectorTransaction> promise;
    auto future = promise.get_future();

    mPromises.push_back(std::move(promise));

    lock.unlock();
    mConditionVariable.notify_all();  // Notify the thread that there are new promises

    return future;
}


void cam::LandmarkDetector::pushFrame(Frame * frame) {
    // if the frame is null, we ignore it
    if( ! frame ) {
        return;
    }

    std::lock_guard<std::mutex> lock(mLock);

    // already exited ?
    if(mExit) {
        Logger::error("exit flag is set");
        return;
    }

    mFrame = std::unique_ptr<Frame>(frame);
}


void cam::LandmarkDetector::run() {

    while(true) {

        std::unique_lock<std::mutex> lock(mLock);

        mConditionVariable.wait(lock, [this] { return mExit || !mPromises.empty(); });

        if(mExit) {
            break;
        }

        // check
        if(mPromises.empty()) {
            Logger::suicide("empty promises list");
        }

        // list of promises for the same current frame
        std::list<std::promise<TLandmarkDetectorTransaction>> promises = std::move(mPromises);
        mPromises.clear();

        std::unique_ptr<Frame> frame = nullptr;
        if( ! mFrame ) {
            // deepcopy of frame
            frame = std::make_unique<Frame>(*mFrame, true);
        }

        lock.unlock();


        bool predict = true;
        TLandmarkDetectorTransaction returnValue = TLandmarkDetectorTransaction();
        returnValue.mSuccess = false;

        // if A) or B) -> we set all promises's futures to failed (they all failed for the current frame)
        // A) if frame is still NULL, we don't have any frame to process
        if( ! frame ) {
            returnValue.mMessage = "cross detector has not yet received a frame from a producer";
            predict = false;
        }
        else {
            // B) if the frame is too old, or there's a time mismatch, we don't use it
            auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
            uint64_t nowTime = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
            uint64_t frameTime = frame->getTimestampRTC();
            if(nowTime < frameTime) {
                // frame has time in the future from now
                // -> this could happen if pc updated its time in between. Don't error, process the promise + message
                returnValue.mMessage = "now time is lower than frame time, pc probably updated time";
                predict = false;
            } else if(nowTime - frameTime > mTimeout_ms) {
                returnValue.mMessage = "frame is older than timeout from current time, so we won't use it";
                predict = false;
            }
        }


        if(predict) {
            // make prediction from current frame
            returnValue = this->makePrediction(frame, promises);
        }


        // set the result for all the current promises
        // -> same result since they are all for the same frame
        for(auto & promise : promises) {
            promise.set_value(returnValue);
        }

    }

}


cam::TLandmarkDetectorTransaction cam::LandmarkDetector::makePrediction(std::unique_ptr<Frame> frame, std::list<std::promise<TLandmarkDetectorTransaction>> & promises) {
    if( ! frame ) {
        Logger::error("frame is NULL");
        return false;
    }
    if(promises.empty()) {
        Logger::error("empty promises");
        return false;
    }

    TLandmarkDetectorTransaction returnValue = TLandmarkDetectorTransaction();

    // ------------------
    // preprocess image
    // ------------------
    cv::Mat input;
    if( ! this->preprocess(frame, input) ) {
        returnValue.mSuccess = false;
        returnValue.mMessage = "failed to preprocess frame before passing it as input to the model";
        return returnValue;
    }

    // ------------------
    // prediction - forward pass
    // ------------------
    try {
        std::unique_lock<std::mutex> lock(mModelLock);

        // Set the input blob
        mModel.setInput(input);

        // Run forward pass
        cv::Mat classOutput = mModel.forward("class");
        cv::Mat bboxOutput = mModel.forward("bounding_box");

        lock.unlock();

        // Extract output values
        float crossProbability = classOutput.at<float>(0, 0);
        uint16_t crossXCenter = static_cast<uint16_t>(std::round( bboxOutput.at<float>(0, 0) * frame->getWidth() ));
        uint16_t crossYCenter = static_cast<uint16_t>(std::round( bboxOutput.at<float>(0, 1) * frame->getHeight() ));
        uint16_t crossWidth = static_cast<uint16_t>(std::round( bboxOutput.at<float>(0, 2) * frame->getWidth() ));
        uint16_t crossHeight = static_cast<uint16_t>(std::round( bboxOutput.at<float>(0, 3) * frame->getHeight() ));

        returnValue = TLandmarkDetectorTransaction(true, "", crossProbability, crossXCenter, crossYCenter, crossWidth, crossHeight);

    } catch( cv::Exception & e ) {
        Logger::message(std::format("cross detection prediction exception caught: {}", e.what()));
        returnValue.mSuccess = false;
        returnValue.mMessage = "failed to make prediction";
    }

    return returnValue;
}


bool cam::LandmarkDetector::preprocess(std::unique_ptr<Frame>, cv::Mat & output) {
    if( ! frame ) {
        Logger::error("frame is NULL");
        return false;
    }
    // 1. convert to cv::Mat
    char pixelSize = 0;
    if(frame->getPixelFormat() == Frame::EPixelFormat::PIXEL_FORMAT_GRAY8)
        pixelSize = 1;
    else if(frame->getPixelFormat() == Frame::EPixelFormat::PIXEL_FORMAT_RGB24)
        pixelSize = 3;
    else {
        Logger::error(std::format("unknown pixel format {}", static_cast<int>(frame->getPixelFormat())));
        return false;
    }

    int type = 0;
    if(pixelSize == 1)
        type = CV_8UC1;
    else if(pixelSize == 3)
        type = CV_8UC3;
    else {
        Logger::error(std::format("unknown pixel size {}", pixelSize));
        return false;
    }

    try {
        size_t unpaddedWidth = static_cast<size_t>(frame->getWidth()) * static_cast<size_t>(pixelSize);
        size_t step = unpaddedWidth + static_cast<size_t>(frame->getPaddingX());
        cv::Mat input = cv::Mat(frame->getHeight(), frame->getWidth(), type, frame->getBuffer(), step);

        // 2. convert BGR to Gray
        if(pixelSize == 3) {
            cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);
        }

        // 3. resize the image to match the model's input size
        cv::resize(input, input, cv::Size(mModelInputWidth, mModelInputHeight));

        // 4. normalise pixels
        input.convertTo(input, CV_32F);
        input /= 255.0;

        // 5. stretch image to force a minimum of 0 and maximum of 1 for all channels, removing intensity variations
        double pixel_min, pixel_max;
        cv::minMaxLoc(input, &pixel_min, &pixel_max);

        double a = 1.0 / (pixel_max - pixel_min);
        double b = -pixel_min * a;

        input = a * input + b;

        // 6. convert grayscale image to rgb by repeating the channel three times
        cv::cvtColor(input, input, cv::COLOR_GRAY2BGR);

        // 7. convert input to NCHW dimension array, as expected by the model
        output = cv::dnn::blobFromImage(input, 1.0, cv::Size(mModelInputWidth, mModelInputHeight), cv::Scalar(0, 0, 0), false, false);
    } catch( cv::Exception & e ) {
        Logger::message(std::format("cross detection preprocess exception caught: {}", e.what()));
        return false;
    }

    return true;
}