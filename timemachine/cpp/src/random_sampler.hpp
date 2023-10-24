#pragma once

#include <memory>
#include <vector>

#include "curand.h"
#include "device_buffer.hpp"
#include <cub/util_type.cuh>

namespace timemachine {

// An RandomSampler for random sampling from probabilities (np.random.choice(X, replace=True p=probabilities))
template <typename RealType> class RandomSampler {

private:
    const int N_; // Max number of values that can be sampled from
    size_t temp_storage_bytes_;

    DeviceBuffer<RealType> d_rand_;
    DeviceBuffer<RealType> d_gumbel_;
    DeviceBuffer<cub::KeyValuePair<int, RealType>> d_arg_max_;

    std::unique_ptr<DeviceBuffer<char>> d_sort_storage_;

    curandGenerator_t cr_rng_;

public:
    RandomSampler(const int N, const int seed);

    ~RandomSampler();

    void
    sample_device(const int N, const int K, const RealType *d_log_probabilities, int *d_samples, cudaStream_t stream);

    std::vector<int> sample_host(const int K, const std::vector<RealType> &probabilities);
};

} // namespace timemachine
