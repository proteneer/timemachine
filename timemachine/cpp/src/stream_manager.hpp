// this implements a stream manager that allows creation of new streams as well as syncing two streams.
// Handles the creation and destruction of streams and events, of which stream destruction is blocking while event destruction is not.
// Streams are all created to not sync with the NULL stream and events have timings disabled.
#pragma once

#include <map>

namespace timemachine {

class StreamManager {

public:
    StreamManager();

    ~StreamManager();

    // get_stream handles the creation and retrieval of cuda streams. Streams are configured
    // to not sync implicitly with the NULL stream.
    cudaStream_t get_stream(int key);

    // get_event handles the creation and retrieval of cuda events. The events have timings disabled for performance.
    cudaEvent_t get_stream_event(int key);

    // sync_to will sync the to_stream stream with the stream associated with the key. This is done on the GPU and does not block.
    void sync_to(int key, cudaStream_t to_stream);

    // sync_from will sync the stream associated with the key to the from_stream. This is done on the GPU and does not block.
    void sync_from(int key, cudaStream_t from_stream);

private:
    std::map<int, cudaStream_t> streams_;
    std::map<int, cudaEvent_t> events_;
};

} // namespace timemachine
