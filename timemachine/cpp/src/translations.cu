#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_translations.cuh"
#include "math_utils.cuh"
#include "translations.hpp"

namespace timemachine {

template <typename RealType>
std::vector<RealType> get_translations_within_sphere_host(
    const int n_translations, const std::vector<RealType> &center, const RealType radius, const int seed) {

    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    DeviceBuffer<int> d_inner(1);
    int flag = 1;
    d_inner.copy_from(&flag);
    DeviceBuffer<double> d_box(3 * 3); // Dummy value, not important
    DeviceBuffer<RealType> d_center(center);
    DeviceBuffer<curandState_t> d_states(tpb);
    DeviceBuffer<RealType> d_output(n_translations * 3);

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    k_initialize_curand_states<<<1, tpb, 0, stream>>>(tpb, seed, d_states.data);
    gpuErrchk(cudaPeekAtLastError());

    k_generate_translations_within_or_outside_a_sphere<RealType><<<ceil_divide(n_translations, tpb), tpb, 0, stream>>>(
        n_translations, d_box.data, d_center.data, d_inner.data, radius, d_states.data, d_output.data);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaStreamSynchronize(stream));

    std::vector<RealType> h_out(d_output.length);
    d_output.copy_to(&h_out[0]);
    return h_out;
}

template std::vector<float>
get_translations_within_sphere_host<float>(const int, const std::vector<float> &center, const float, const int);
template std::vector<double>
get_translations_within_sphere_host<double>(const int, const std::vector<double> &center, const double, const int);

template <typename RealType>
std::vector<RealType> get_translations_outside_sphere_host(
    const int n_translations,
    const std::vector<double> &box,
    const std::vector<RealType> &center,
    const RealType radius,
    const int seed) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;

    DeviceBuffer<int> d_inner(1);
    int flag = 0;
    d_inner.copy_from(&flag);
    DeviceBuffer<double> d_box(box);
    DeviceBuffer<RealType> d_center(center);
    DeviceBuffer<curandState_t> d_states(tpb);
    DeviceBuffer<RealType> d_output(n_translations * 3);

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    k_initialize_curand_states<<<1, tpb, 0, stream>>>(tpb, seed, d_states.data);
    gpuErrchk(cudaPeekAtLastError());

    k_generate_translations_within_or_outside_a_sphere<RealType><<<ceil_divide(n_translations, tpb), tpb, 0, stream>>>(
        n_translations, d_box.data, d_center.data, d_inner.data, radius, d_states.data, d_output.data);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaStreamSynchronize(stream));

    std::vector<RealType> h_out(d_output.length);
    d_output.copy_to(&h_out[0]);
    return h_out;
}

template std::vector<float> get_translations_outside_sphere_host<float>(
    const int, const std::vector<double> &box, const std::vector<float> &center, const float, const int);
template std::vector<double> get_translations_outside_sphere_host<double>(
    const int, const std::vector<double> &box, const std::vector<double> &center, const double, const int);

} // namespace timemachine
