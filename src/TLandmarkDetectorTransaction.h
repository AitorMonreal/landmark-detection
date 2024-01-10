#ifndef TLANDMARKDETECTORTRANSACTION_H
#define TLANDMARKDETECTORTRANSACTION_H

#include <stdint.h>
#include <string>


namespace cam {

    struct TLandmarkDetectorTransaction {

        bool mSuccess;
        std::string mMessage;

        float mCrossProbability;
        uint16_t mCrossCenterX;
        uint16_t mCrossCenterY;
        uint16_t mCrossWidth;
        uint16_t mCrossHeight;

        TLandmarkDetectorTransaction() : mSuccess(false), mMessage(std::string()), mCrossProbability(0.0), mCrossCenterX(0), mCrossCenterY(0), mCrossWidth(0), mCrossHeight(0) {}
        TLandmarkDetectorTransaction(bool success, std::string message, float crossProbability, uint16_t crossCenterX, uint16_t crossCenterY, uint16_t crossWidth, uint16_t crossHeight) :
            mSuccess(success), mMessage(message), mCrossProbability(crossProbability), mCrossCenterX(crossCenterX), mCrossCenterY(crossCenterY), mCrossWidth(crossWidth), mCrossHeight(crossHeight) {}
    };

}

#endif // TLANDMARKDETECTORTRANSACTION_H
