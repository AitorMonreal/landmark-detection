#ifndef LANDMARKDETECTOR_H
#define LANDMARKDETECTOR_H

#include <stdint.h>
#include <string>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>
#include <format>

#include "Frame.h"
#include "StreamConsumer.h"

#include "TLandmarkDetectorTransaction.h"

#include "logger.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"


namespace cam {

    class LandmarkDetector : public StreamConsumer {

        public:

            LandmarkDetector();
            virtual ~LandmarkDetector();

            // no copy, no copy assignment
            LandmarkDetector(const LandmarkDetector& other) = delete;
            LandmarkDetector& operator=(const LandmarkDetector& other) = delete;


        public:

            void pushFrame(Frame* frame) override;


        public:

            std::future<TLandmarkDetectorTransaction> predict();


        private:

            void run();


        private:

            TLandmarkDetectorTransaction makePrediction(const std::unique_ptr<Frame>& frame);
            bool preprocess(const std::unique_ptr<Frame>& frame, cv::Mat& output);


        private:

            std::thread mThread;

            std::mutex mLock;
            std::condition_variable mConditionVariable;

            std::mutex mModelLock;

            bool mExit;

            std::list<std::promise<TLandmarkDetectorTransaction>> mPromises;

            std::unique_ptr<Frame> mFrame;

            cv::dnn::Net mModel;
            static constexpr uint16_t mModelInputWidth = 256;
            static constexpr uint16_t mModelInputHeight = 224;

            static constexpr uint64_t mTimeout_ms = 7ul*1000ul;  // milliseconds
            static constexpr char mModelFilePath[] = ".../detection_model.onnx";
        };
}

#endif // LANDMARKDETECTOR_H