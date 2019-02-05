#include "bonded_cpu.hpp"


int main() {
    std::vector<double> params({5.0, 1.0, 2.2, 3.0});
    std::vector<size_t> param_idxs({
        1, 3,
        0, 2,
        0, 1,
        1, 2
    });
    std::vector<size_t> bond_idxs({
        0, 1,
        0, 2,
        0, 3,
        0, 4
    });
    
    timemachine::HarmonicBond<double> hb(
        params,
        param_idxs,
        bond_idxs
    );
    std::vector<double> geometry({
         0.0637,   0.0126,   0.2203,
         1.0573,  -0.2011,   1.2864,
         2.3928,   1.2209,  -0.2230,
        -0.6891,   1.6983,   0.0780,
        -0.6312,  -1.6261,  -0.2601
    });
    std::vector<double> dxdp({
        0.10422066, 0.17122579, 0.72177294,
        0.08101986, 0.95524382, 0.77068016,
        0.3582338 , 0.70634414, 0.29395797,
        0.89439972, 0.23706983, 0.01236061,
        0.95828988, 0.49409002, 0.72811251,

        0.6789271 , 0.84382456, 0.4675572,
        0.12289735, 0.01340453, 0.99346565,
        0.41574403, 0.89688291, 0.02852552,
        0.83768939, 0.99084417, 0.87983468,
        0.15627564, 0.18022404, 0.4307694
    });

    std::vector<double> grad_out(5*3, 0.0);
    std::vector<double> total_out(2*5*3, 0.0);
    double energy = 0;

    hb.total_derivative(
        5,
        2,
        &geometry[0],
        &dxdp[0],
        &energy,
        &grad_out[0],
        &total_out[0]
    );

    for(auto i=0; i < 2*5*3; i++) {
        std::cout << total_out[i] << std::endl;
    }

}