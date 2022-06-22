
template <typename RealType> __device__ struct Matrix { RealType x0, y0, z0, x1, y1, z1, x2, y2, z2; };

template <typename RealType> __device__ struct Vector {

    RealType x, y, z;

    __device__ RealType dot(Vector<RealType> const &other) { return x * other.x + y * other.y + z * other.z; }

    __device__ RealType norm() const { return std::sqrt(x * x + y * y + z * z); }

    __device__ Vector<RealType> unit() const {
        RealType n = norm();
        return {x / n, y / n, z / n};
    }

    __device__ Matrix<RealType> unit_jac() const {
        RealType n = this->norm();
        Vector<RealType> v = this->unit();
        return {
            (1 - v.x * v.x) / n,
            (-v.x * v.y) / n,
            (-v.x * v.z) / n,
            (-v.y * v.x) / n,
            (1 - v.y * v.y) / n,
            (-v.y * v.z) / n,
            (-v.z * v.x) / n,
            (-v.z * v.y) / n,
            (1 - v.z * v.z) / n};
    }

    __device__ Vector<RealType> operator+(Vector<RealType> const &obj) const {
        return {x + obj.x, y + obj.y, z + obj.z};
    }

    __device__ Vector<RealType> operator-(Vector<RealType> const &obj) const {
        return {x - obj.x, y - obj.y, z - obj.z};
    }

    // unary minus
    __device__ Vector<RealType> operator-() const { return {-x, -y, -z}; }
};

template <typename RealType> __device__ Vector<RealType> mul_vm(Vector<RealType> v, Matrix<RealType> m) {

    RealType v0 = v.x * m.x0 + v.y * m.x1 + v.z * m.x2;
    RealType v1 = v.x * m.y0 + v.y * m.y1 + v.z * m.y2;
    RealType v2 = v.x * m.z0 + v.y * m.z1 + v.z * m.z2;

    return Vector<RealType>({v0, v1, v2});
}

template <typename RealType> __device__ Vector<RealType> cross_product(Vector<RealType> a, Vector<RealType> b) {

    RealType cx = a.y * b.z - a.z * b.y;
    RealType cy = a.z * b.x - a.x * b.z;
    RealType cz = a.x * b.y - a.y * b.x;

    return Vector<RealType>({cx, cy, cz});
}

template <typename RealType>
__device__ void cross_jac(Vector<RealType> a, Vector<RealType> b, Matrix<RealType> &jac_a, Matrix<RealType> &jac_b) {

    // clang-format off
    jac_a.x0 = 0; jac_a.y0 = b.z; jac_a.z0 = -b.y;
    jac_a.x1 = -b.z; jac_a.y1 = 0; jac_a.z1 = b.x;
    jac_a.x2 = b.y; jac_a.y2 = -b.x; jac_a.z2 = 0;

    jac_b.x0 = 0; jac_b.y0 = -a.z; jac_b.z0 = a.y;
    jac_b.x1 = a.z; jac_b.y1 = 0; jac_b.z1 = -a.x;
    jac_b.x2 = -a.y; jac_b.y2 = a.x; jac_b.z2 = 0;
    // clang-format on
}

// template<typename RealType>
// void print_matrix(Matrix<RealType> m) {
//     std::cout << "[" << std::endl;
//     std::cout << m.x0 << " " << m.y0 << " " << m.z0 << std::endl;
//     std::cout << m.x1 << " " << m.y1 << " " << m.z1 << std::endl;
//     std::cout << m.x2 << " " << m.y2 << " " << m.z2 << std::endl;
//     std::cout << "]" << std::endl;
// }

template <typename RealType>
__device__ void pyramidal_vol_and_grad(
    Vector<RealType> xc,
    Vector<RealType> x1,
    Vector<RealType> x2,
    Vector<RealType> x3,
    RealType &vol,
    Vector<RealType> &xc_grad,
    Vector<RealType> &x1_grad,
    Vector<RealType> &x2_grad,
    Vector<RealType> &x3_grad) {

    auto xx = x1 - xc;
    auto yy = x2 - xc;
    auto zz = x3 - xc;

    auto x = xx.unit();
    auto y = yy.unit();
    auto z = zz.unit();

    auto xy = cross_product(x, y);

    Matrix<RealType> dc_dx;
    Matrix<RealType> dc_dy;

    cross_jac(x, y, dc_dx, dc_dy);

    auto dx_dxx = xx.unit_jac();
    auto dy_dyy = yy.unit_jac();
    auto dz_dzz = zz.unit_jac();

    auto do_dxx = mul_vm(mul_vm(z, dc_dx), dx_dxx);
    auto do_dyy = mul_vm(mul_vm(z, dc_dy), dy_dyy);
    auto do_dzz = mul_vm(xy, dz_dzz);

    vol = xy.dot(z);

    xc_grad = -do_dxx - do_dyy - do_dzz;
    x1_grad = do_dxx;
    x2_grad = do_dyy;
    x3_grad = do_dzz;
}

template <typename RealType>
__device__ void torsion_vol_and_grad(
    Vector<RealType> x0,
    Vector<RealType> x1,
    Vector<RealType> x2,
    Vector<RealType> x3,
    RealType &vol,
    Vector<RealType> &x0_grad,
    Vector<RealType> &x1_grad,
    Vector<RealType> &x2_grad,
    Vector<RealType> &x3_grad) {

    auto xx = x1 - x0;
    auto yy = x1 - x2;
    auto zz = x3 - x2;

    auto x = xx.unit();
    auto y = yy.unit();
    auto z = zz.unit();

    auto xy = cross_product(x, y);
    auto yz = cross_product(y, z);

    Matrix<RealType> dc0_dx;
    Matrix<RealType> dc0_dy;
    cross_jac(x, y, dc0_dx, dc0_dy);

    Matrix<RealType> dc1_dy;
    Matrix<RealType> dc1_dz;
    cross_jac(y, z, dc1_dy, dc1_dz);

    auto do_dc0 = yz;
    auto do_dc1 = xy;

    auto dx_dxx = xx.unit_jac();
    auto dy_dyy = yy.unit_jac();
    auto dz_dzz = zz.unit_jac();

    auto do_dxx = mul_vm(mul_vm(do_dc0, dc0_dx), dx_dxx);
    auto do_dyy = mul_vm(mul_vm(do_dc0, dc0_dy), dy_dyy) + mul_vm(mul_vm(do_dc1, dc1_dy), dy_dyy);
    auto do_dzz = mul_vm(mul_vm(do_dc1, dc1_dz), dz_dzz);

    vol = xy.dot(yz);

    x0_grad = -do_dxx;
    x1_grad = do_dxx + do_dyy;
    x2_grad = -do_dyy - do_dzz;
    x3_grad = do_dzz;
}
