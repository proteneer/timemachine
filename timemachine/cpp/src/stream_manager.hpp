// this implements a runner for running potentials in multiple streams and then syncing
// with the parent stream
#pragma once

#include <map>

namespace timemachine {

class StreamManager {

public:
    StreamManager();

    ~StreamManager();

    cudaStream_t get_stream(int key);
    cudaEvent_t get_stream_event(int key);

    void sync_to(int key, cudaStream_t to_stream);
    void sync_from(int key, cudaStream_t from_stream);

private:
    std::map<int, cudaStream_t> streams_;
    std::map<int, cudaEvent_t> events_;
};

} // namespace timemachine
