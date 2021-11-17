#include "rmsd_align.hpp"
#include <Eigen/Dense>

#include <iostream> // delete me

namespace timemachine {

/*
Optimally align x2 onto x1. In particular, x2 is shifted so that its centroid is placed
at the same position as the of x1's centroid. x2 is also rotated so that the RMSD
is minimized.
*/
void rmsd_align_cpu(const int N, const double *x1_raw, const double *x2_raw, double *x2_aligned_raw) {

    Eigen::MatrixXd x1(N, 3);
    Eigen::MatrixXd x2(N, 3);

    for (int i = 0; i < N; i++) {
        x1(i, 0) = x1_raw[i * 3 + 0];
        x1(i, 1) = x1_raw[i * 3 + 1];
        x1(i, 2) = x1_raw[i * 3 + 2];

        x2(i, 0) = x2_raw[i * 3 + 0];
        x2(i, 1) = x2_raw[i * 3 + 1];
        x2(i, 2) = x2_raw[i * 3 + 2];
    }

    Eigen::Vector3d x1_centroid = x1.colwise().mean();
    Eigen::Vector3d x2_centroid = x2.colwise().mean();
    Eigen::Vector3d translation = x2_centroid - x1_centroid;

    // shift to the center
    Eigen::MatrixXd x1_centered = x1.rowwise() - x1_centroid.transpose();
    Eigen::MatrixXd x2_centered = x2.rowwise() - x2_centroid.transpose();

    // compute correlations
    Eigen::MatrixXd c = x2_centered.transpose() * x1_centered;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(c, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto s = svd.singularValues();

    Eigen::MatrixXd u = svd.matrixU();
    Eigen::MatrixXd v = svd.matrixV();
    Eigen::MatrixXd v_t = v.transpose();

    bool is_reflection = u.determinant() * v_t.determinant() < 0.0;
    if (is_reflection) {
        for (int i = 0; i < 3; i++) {
            u(i, 2) = -u(i, 2);
        }
    }

    Eigen::MatrixXd rotation = u * v_t;

    // x2 is centered
    Eigen::MatrixXd x2_rot = x2_centered * rotation;
    Eigen::MatrixXd x2_aligned = x2_rot.rowwise() - (translation.transpose() - x2_centroid.transpose());

    for (int i = 0; i < N; i++) {
        x2_aligned_raw[i * 3 + 0] = x2_aligned(i, 0);
        x2_aligned_raw[i * 3 + 1] = x2_aligned(i, 1);
        x2_aligned_raw[i * 3 + 2] = x2_aligned(i, 2);
    }
}

} // namespace timemachine
