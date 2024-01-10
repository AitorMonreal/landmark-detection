
#ifndef STREAMCONSUMER_H
#define STREAMCONSUMER_H

#include "Frame.h"


namespace cam {

    class StreamConsumer {

        public:

            StreamConsumer() = default;
            virtual ~StreamConsumer();

            StreamConsumer(const StreamConsumer & other) = delete;
            StreamConsumer& operator=(const StreamConsumer & other) = delete;

            virtual void pushFrame(Frame * frame) = 0;

    };
}

#endif  // STREAMCONSUMER_H
