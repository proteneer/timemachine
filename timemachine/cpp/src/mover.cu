#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include "mover.hpp"

namespace timemachine {

std::array<std::vector<double>, 2> Mover::move_host(const int N, const double *h_x, const double *h_box) {

    DeviceBuffer<double> d_x(N * 3);
    DeviceBuffer<double> d_box(3 * 3);
    d_x.copy_from(h_x);
    d_box.copy_from(h_box);

    cudaStream_t stream = static_cast<cudaStream_t>(0);

    this->move(N, d_x.data, d_box.data, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
    std::vector<double> out_coords(d_x.length);
    std::vector<double> out_box(d_box.length);
    d_x.copy_to(&out_coords[0]);
    d_box.copy_to(&out_box[0]);
    return std::array<std::vector<double>, 2>({out_coords, out_box});
}

} // namespace timemachine
