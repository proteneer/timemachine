#pragma once

#include "kernel_utils.cuh"
#define ERR (1.e-24)

template<typename RealType>
struct Surreal {
    // __device__
    RealType real, imag;

    __device__ Surreal<RealType>(const RealType& v=0.0, const RealType& d=0.0) : real(v), imag(d) {}
    __device__ inline Surreal& operator=(const Surreal & s) {real = s.real ; imag = s.imag; return *this;};

    __device__ inline Surreal operator+(const Surreal&) const;
    __device__ inline Surreal operator+(const RealType&) const;
    __device__ inline Surreal& operator+=(const Surreal&);
    __device__ inline Surreal& operator+=(const RealType&);

    __device__ inline Surreal operator-() const;
    __device__ inline Surreal operator-(const Surreal&) const;
    __device__ inline Surreal operator-(const RealType&) const;

    __device__ inline Surreal& operator-=(const Surreal&);
    __device__ inline Surreal& operator-=(const RealType&);

    __device__ inline Surreal operator*(const Surreal&) const;
    __device__ inline Surreal operator*(const RealType&) const;

    __device__ inline Surreal& operator*=(const Surreal&);
    __device__ inline Surreal& operator*=(const RealType&);

    __device__ inline Surreal operator/(const Surreal&) const;
    __device__ inline Surreal operator/(const RealType&) const;

    __device__ inline Surreal& operator/=(const Surreal&);
    __device__ inline Surreal& operator/=(const RealType&);
    __device__ inline Surreal& operator/=(const int&);

    __device__ inline Surreal sin(const Surreal&);
    __device__ inline Surreal cos(const Surreal&);

    __device__ inline Surreal acos(const Surreal&);
    __device__ inline Surreal atan2(const Surreal&, const Surreal&);

    __device__ inline Surreal sqrt(const Surreal&);
    __device__ inline Surreal exp(const Surreal&);

};

template <typename RealType>
__device__ inline Surreal<RealType> Surreal<RealType>::operator+(const Surreal<RealType>& z) const {
    return Surreal<RealType>(real+z.real,imag+z.imag);
}

template <typename RealType>
__device__ inline Surreal<RealType> Surreal<RealType>::operator+(const RealType& r) const {
    return Surreal<RealType>(real+r,imag);
}

template <typename RealType>
__device__ inline Surreal<RealType>& Surreal<RealType>::operator+=(const Surreal<RealType>& z) {
    real+=z.real;
    imag+=z.imag;
    return *this;
}

template <typename RealType>
__device__ inline Surreal<RealType>& Surreal<RealType>::operator+=(const RealType& r) {
    real+=r;
    return *this;
}

template <typename RealType>
__device__ inline Surreal<RealType> operator+(const RealType& r, const Surreal<RealType>& z) {
    return Surreal<RealType>(r+z.real,z.imag);
}

template <typename RealType>
__device__ inline Surreal<RealType> operator+(const int& i, const Surreal<RealType>& z) {
    return Surreal<RealType>(RealType(i)+z.real,z.imag);
}

template <typename RealType>
__device__ inline Surreal<RealType> Surreal<RealType>::operator-() const {
    return Surreal<RealType>(-real,-imag);
}

template <typename RealType>
__device__ inline Surreal<RealType> Surreal<RealType>::operator-(const Surreal<RealType>& z) const {
    return Surreal<RealType>(real-z.real,imag-z.imag);
}

template <typename RealType>
__device__ inline Surreal<RealType> Surreal<RealType>::operator-(const RealType& r) const {
    return Surreal<RealType>(real-r,imag);
}

template <typename RealType>
__device__ inline Surreal<RealType>& Surreal<RealType>::operator-=(const Surreal<RealType>& z) {
    real -= z.real;
    imag -= z.imag;
    return *this;
}

template <typename RealType>
__device__ inline Surreal<RealType>& Surreal<RealType>::operator-=(const RealType& r) {
    real -= r;
    return *this;
}

template <typename RealType>
__device__ inline Surreal<RealType> operator-(const RealType& r, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        r-z.real,
        -z.imag
    );
}

template <typename RealType>
__device__ inline Surreal<RealType> Surreal<RealType>::operator*(const Surreal<RealType>& z) const {
    return Surreal<RealType>(
        real*z.real,
        real*z.imag+z.real*imag
    );
}

template <typename RealType>
__device__ inline Surreal<RealType> Surreal<RealType>::operator*(const RealType& r) const {
    return Surreal<RealType>(
        real*r,
        imag*r
    );
}

template <typename RealType>
__device__ inline Surreal<RealType>& Surreal<RealType>::operator*=(const Surreal<RealType>& z) {
    imag = real*z.imag+z.real*imag;
    real *= z.real;
    return *this;
}

template <typename RealType>
__device__ inline Surreal<RealType>& Surreal<RealType>::operator*=(const RealType& r) {
    real *= r;
    imag *= r;
    return *this;
}

template <typename RealType>
__device__ inline Surreal<RealType> operator*(const RealType& r, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        r*z.real,
        r*z.imag
    );
}

template <typename RealType>
__device__ inline Surreal<RealType> operator*(const int& i, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        RealType(i)*z.real,
        RealType(i)*z.imag
    );
}

template <typename RealType>
__device__ inline Surreal<RealType> Surreal<RealType>::operator/(const Surreal<RealType>& z) const {
    return Surreal<RealType>(
        real/z.real,
        (z.real*imag-real*z.imag)/(z.real*z.real)
    );
}

template <typename RealType>
__device__ inline Surreal<RealType> Surreal<RealType>::operator/(const RealType& r) const {
    return Surreal<RealType>(real/r,imag/r);
}

template <typename RealType>
__device__ inline Surreal<RealType>& Surreal<RealType>::operator/=(const Surreal<RealType>& z) {
    imag = (z.real*imag-real*z.imag)/(z.real*z.real);
    real /= z.real;
    return *this;
}

template <typename RealType>
__device__ inline Surreal<RealType>& Surreal<RealType>::operator/=(const RealType& r) {
    real /= r;
    imag /= r;
    return *this;
}

template <typename RealType>
__device__ inline Surreal<RealType>& Surreal<RealType>::operator/=(const int& i) {
    real /= RealType(i);
    real /= RealType(i);
    return *this;
}

template <typename RealType>
__device__ inline Surreal<RealType> operator/(const RealType& r, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        r/z.real,
        -r*z.imag/(z.real*z.real)
    );
}

template <typename RealType>
__device__ inline Surreal<RealType> operator/(const int& i, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        RealType(i)/z.real,
        -RealType(i)*z.imag/(z.real*z.real)
    );
}

template <typename RealType>
__device__ inline Surreal<RealType> sin(const Surreal<RealType>& z) {
    return Surreal<RealType>(
        sin(z.real),
        z.imag*cos(z.real)
    );
}

template <typename RealType>
__device__ inline Surreal<RealType> cos(const Surreal<RealType>& z) {
    return Surreal<RealType>(
        cos(z.real),
        -z.imag*sin(z.real)
    );
}

template <typename RealType>
__device__ inline Surreal<RealType> acos(const Surreal<RealType>& z) {
    return Surreal<RealType>(
        acos(z.real),
        -z.imag/sqrt(1.0-z.real*z.real+ERR)
    );
}

template <typename RealType>
__device__ inline Surreal<RealType> atan2(const Surreal<RealType>& z1, const Surreal<RealType>& z2) {
    return Surreal<RealType>(
        atan2(z1.real,z2.real),
        (z2.real*z1.imag-z1.real*z2.imag)/(z1.real*z1.real+z2.real*z2.real)
    );
}

template <typename RealType>
__device__ inline Surreal<RealType> sqrt(const Surreal<RealType>& z) {
    RealType sqrtv = gpuSqrt(z.real);
    return Surreal<RealType>(
        sqrtv,
        0.5*z.imag/(sqrtv+ERR)
    );
}

template <typename RealType>
__device__ inline Surreal<RealType> exp(const Surreal<RealType>& z) {
    RealType expv = exp(z.real);
    return Surreal<RealType>(
        expv,
        z.imag*expv
    );
}