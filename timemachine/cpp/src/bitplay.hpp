#pragma once

namespace timemachine {

class BitPlay {

public:

    BitPlay(
        int N,
        const double *x_0
    );

    ~BitPlay();

    void compute_dist();
    void compute_dist_f32();

    void compute_dist_tensor();

    void get_dist(float *out_buffer);
    int N_; // number of particles

private:

    float *d_dist_;
    double *d_x_t_; // coordinates

};

}

