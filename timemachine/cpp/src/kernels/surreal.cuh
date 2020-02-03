#pragma once

// #include "kernel_utils.cuh"
#define ERR (1.e-24)
#ifdef __NVCC__
    #define DECL __host__ __device__
#else
    #define DECL
#endif

template<typename RealType>
struct Surreal {

    typedef RealType value_type;
    RealType real, imag;

    DECL Surreal<RealType>(const RealType& v=0.0, const RealType& d=0.0) : real(v), imag(d) {}

    DECL Surreal& operator=(const Surreal & s) {real = s.real ; imag = s.imag; return *this;};

    DECL Surreal operator+(const Surreal&) const;
    DECL Surreal operator+(const RealType&) const;

    DECL Surreal& operator+=(const Surreal&);
    DECL Surreal& operator+=(const RealType&);

    DECL Surreal operator-() const;
    DECL Surreal operator-(const Surreal&) const;
    DECL Surreal operator-(const RealType&) const;

    DECL Surreal& operator-=(const Surreal&);
    DECL Surreal& operator-=(const RealType&);

    DECL Surreal operator*(const Surreal&) const;
    DECL Surreal operator*(const RealType&) const;

    DECL Surreal& operator*=(const Surreal&);
    DECL Surreal& operator*=(const RealType&);

    DECL Surreal operator/(const Surreal&) const;
    DECL Surreal operator/(const RealType&) const;

    DECL Surreal& operator/=(const Surreal&);
    DECL Surreal& operator/=(const RealType&);
    DECL Surreal& operator/=(const int&);

    DECL Surreal sin(const Surreal&);
    DECL Surreal cos(const Surreal&);

    DECL Surreal acos(const Surreal&);
    DECL Surreal atan2(const Surreal&, const Surreal&);

    DECL Surreal sqrt(const Surreal&);
    DECL Surreal exp(const Surreal&);

};

template <typename RealType>
DECL Surreal<RealType> abs(const Surreal<RealType>& v) {
    return v.real < 0.0 ? -v: v;
}

template <typename RealType>
DECL Surreal<RealType> max(const double &a, const Surreal<RealType>& v) {
    return v.real > a ? v: Surreal<RealType>(a);
}

#ifdef __NVCC__
template <typename RealType>
DECL Surreal<RealType> __shfl_sync(unsigned mask, Surreal<RealType> &var, int srcLane, int width=warpSize) {
    var.real = __shfl_sync(mask, var.real, srcLane, width);
    var.imag = __shfl_sync(mask, var.imag, srcLane, width);
    return var;
}

template <typename RealType>
__device__ inline void atomicAddOffset(Surreal<RealType> *base_ptr, const unsigned offset, const Surreal<RealType> &val) {
    RealType* real_ptr = reinterpret_cast<RealType*>(base_ptr) + offset*2;
    RealType* imag_ptr = real_ptr + 1;
    atomicAdd(real_ptr, val.real);
    atomicAdd(imag_ptr, val.imag);
}

#endif

template <typename RealType>
DECL bool operator>(const Surreal<RealType> &l, const Surreal<RealType> &r) {
    return l.real > r.real;
}

template <typename RealType>
DECL bool operator<(const Surreal<RealType> &l, const Surreal<RealType> &r) {
    return l.real < r.real;
}

template <typename RealType>
DECL bool operator>(const Surreal<RealType> &l, const double &r) {
    return l.real > r;
}

template <typename RealType>
DECL bool operator<(const Surreal<RealType> &l, const double &r) {
    return l.real < r;
}

// template <typename RealType>
// __host__ __device__ inline RealType real_part(const Surreal<RealType> &a) {
//     return a.real;
// }

// template <typename RealType>
// DECL bool operator>(const Surreal<RealType> &l, const float& r) {
//     return l.real > r;
// }

// template <typename RealType>
// DECL bool operator<(const Surreal<RealType> &l, const double& r) {
//     return l.real < r;
// }


template <typename RealType>
DECL void add_square(Surreal<RealType> &a, const Surreal<RealType> &b) {
    // (ytz): do *not* change the order of operations here.

    a.imag += 2*b.real*b.imag;
    a.real += b.real*b.real;
    // return a;
}



template <typename RealType>
DECL Surreal<RealType> Surreal<RealType>::operator+(const Surreal<RealType>& z) const {
    return Surreal<RealType>(real+z.real,imag+z.imag);
}

template <typename RealType>
DECL Surreal<RealType> Surreal<RealType>::operator+(const RealType& r) const {
    return Surreal<RealType>(real+r,imag);
}

template <typename RealType>
DECL Surreal<RealType>& Surreal<RealType>::operator+=(const Surreal<RealType>& z) {
    real+=z.real;
    imag+=z.imag;
    return *this;
}

template <typename RealType>
DECL Surreal<RealType>& Surreal<RealType>::operator+=(const RealType& r) {
    real+=r;
    return *this;
}

template <typename RealType>
DECL Surreal<RealType> operator+(const RealType& r, const Surreal<RealType>& z) {
    return Surreal<RealType>(r+z.real,z.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator+(const int& i, const Surreal<RealType>& z) {
    return Surreal<RealType>(RealType(i)+z.real,z.imag);
}

template <typename RealType>
DECL Surreal<RealType> Surreal<RealType>::operator-() const {
    return Surreal<RealType>(-real,-imag);
}

template <typename RealType>
DECL Surreal<RealType> Surreal<RealType>::operator-(const Surreal<RealType>& z) const {
    return Surreal<RealType>(real-z.real,imag-z.imag);
}

template <typename RealType>
DECL Surreal<RealType> Surreal<RealType>::operator-(const RealType& r) const {
    return Surreal<RealType>(real-r,imag);
}

template <typename RealType>
DECL Surreal<RealType>& Surreal<RealType>::operator-=(const Surreal<RealType>& z) {
    real -= z.real;
    imag -= z.imag;
    return *this;
}

template <typename RealType>
DECL Surreal<RealType>& Surreal<RealType>::operator-=(const RealType& r) {
    real -= r;
    return *this;
}

template <typename RealType>
DECL Surreal<RealType> operator-(const RealType& r, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        r-z.real,
        -z.imag
    );
}

template <typename RealType>
DECL Surreal<RealType> Surreal<RealType>::operator*(const Surreal<RealType>& z) const {
    return Surreal<RealType>(
        real*z.real,
        real*z.imag+z.real*imag
    );
}

template <typename RealType>
DECL Surreal<RealType> Surreal<RealType>::operator*(const RealType& r) const {
    return Surreal<RealType>(
        real*r,
        imag*r
    );
}

template <typename RealType>
DECL Surreal<RealType>& Surreal<RealType>::operator*=(const Surreal<RealType>& z) {
    imag = real*z.imag+z.real*imag;
    real *= z.real;
    return *this;
}

template <typename RealType>
DECL Surreal<RealType>& Surreal<RealType>::operator*=(const RealType& r) {
    real *= r;
    imag *= r;
    return *this;
}

template <typename RealType>
DECL Surreal<RealType> operator*(const RealType& r, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        r*z.real,
        r*z.imag
    );
}

template <typename RealType>
DECL Surreal<RealType> operator*(const int& i, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        RealType(i)*z.real,
        RealType(i)*z.imag
    );
}

template <typename RealType>
DECL Surreal<RealType> Surreal<RealType>::operator/(const Surreal<RealType>& z) const {
    return Surreal<RealType>(
        real/z.real,
        (z.real*imag-real*z.imag)/(z.real*z.real)
    );
}

template <typename RealType>
DECL Surreal<RealType> Surreal<RealType>::operator/(const RealType& r) const {
    return Surreal<RealType>(real/r,imag/r);
}

template <typename RealType>
DECL Surreal<RealType>& Surreal<RealType>::operator/=(const Surreal<RealType>& z) {
    imag = (z.real*imag-real*z.imag)/(z.real*z.real);
    real /= z.real;
    return *this;
}

template <typename RealType>
DECL Surreal<RealType>& Surreal<RealType>::operator/=(const RealType& r) {
    real /= r;
    imag /= r;
    return *this;
}

template <typename RealType>
DECL Surreal<RealType>& Surreal<RealType>::operator/=(const int& i) {
    real /= RealType(i);
    real /= RealType(i);
    return *this;
}

template <typename RealType>
DECL Surreal<RealType> operator/(const RealType& r, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        r/z.real,
        -r*z.imag/(z.real*z.real)
    );
}

template <typename RealType>
DECL Surreal<RealType> operator/(const int& i, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        RealType(i)/z.real,
        -RealType(i)*z.imag/(z.real*z.real)
    );
}

template <typename RealType>
DECL Surreal<RealType> sin(const Surreal<RealType>& z) {
    return Surreal<RealType>(
        sin(z.real),
        z.imag*cos(z.real)
    );
}

template <typename RealType>
DECL Surreal<RealType> cos(const Surreal<RealType>& z) {
    return Surreal<RealType>(
        cos(z.real),
        -z.imag*sin(z.real)
    );
}

template <typename RealType>
DECL Surreal<RealType> acos(const Surreal<RealType>& z) {
    return Surreal<RealType>(
        acos(z.real),
        -z.imag/sqrt(1.0-z.real*z.real+ERR)
    );
}

template <typename RealType>
DECL Surreal<RealType> atan2(const Surreal<RealType>& z1, const Surreal<RealType>& z2) {
    return Surreal<RealType>(
        atan2(z1.real,z2.real),
        (z2.real*z1.imag-z1.real*z2.imag)/(z1.real*z1.real+z2.real*z2.real)
    );
}

template <typename RealType>
DECL Surreal<RealType> sqrt(const Surreal<RealType>& z) {
    RealType sqrtv = sqrt(z.real);

    return Surreal<RealType>(
        sqrtv,
        // 0.5*z.imag/(sqrtv+ERR)
        0.5*z.imag/(sqrtv)
    );
}

template <typename RealType>
DECL Surreal<RealType> exp(const Surreal<RealType>& z) {
    RealType expv = exp(z.real);
    return Surreal<RealType>(
        expv,
        z.imag*expv
    );
}
