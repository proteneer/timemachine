#pragma once

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

    DECL Surreal<RealType>() {}; // uninitialized for efficiency
    DECL Surreal<RealType>(const RealType& v, const RealType& d) : real(v), imag(d) {}
    DECL Surreal<RealType>(const RealType& v) : real(v), imag(0) {}

    // copy constructor
    DECL Surreal<RealType>(const Surreal<RealType> &z) : real(z.real), imag(z.imag) {}

    DECL Surreal& operator=(const Surreal & s) {real = s.real ; imag = s.imag; return *this;};

    DECL Surreal& operator+=(const Surreal&);
    DECL Surreal& operator+=(const RealType&);

    DECL Surreal operator-() const;

    DECL Surreal& operator-=(const Surreal&);
    DECL Surreal& operator-=(const RealType&);

    DECL Surreal& operator*=(const Surreal&);
    DECL Surreal& operator*=(const RealType&);

    DECL Surreal& operator/=(const Surreal&);
    DECL Surreal& operator/=(const RealType&);
    template<typename OtherType>
    DECL Surreal& operator/=(const OtherType&);

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
DECL Surreal<RealType> fabs(const Surreal<RealType>& v) {
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

template <typename PtrType, typename SecondType>
__device__ inline void atomicAddOffsetSplit(Surreal<PtrType> *base_ptr, const unsigned offset, const Surreal<SecondType> &val) {
    PtrType* real_ptr = reinterpret_cast<PtrType*>(base_ptr) + offset*2;
    PtrType* imag_ptr = real_ptr + 1;
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

template <typename RealType>
DECL Surreal<RealType>& Surreal<RealType>::operator+=(const Surreal<RealType>& z) {
    real += z.real;
    imag += z.imag;
    return *this;
}

template <typename RealType>
DECL Surreal<RealType>& Surreal<RealType>::operator+=(const RealType& r) {
    real += r;
    return *this;
}

template <typename RealType>
DECL Surreal<RealType> operator+(const Surreal<RealType>& a, const Surreal<RealType>& b) {
    return Surreal<RealType>(a.real+b.real, a.imag+b.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator+(const int& i, const Surreal<RealType>& z) {
    return Surreal<RealType>(i+z.real,z.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator+(const float& i, const Surreal<RealType>& z) {
    return Surreal<RealType>(i+z.real,z.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator+(const double& i, const Surreal<RealType>& z) {
    return Surreal<RealType>(i+z.real,z.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator+(const Surreal<RealType>& z, const int& i) {
    return Surreal<RealType>(i+z.real,z.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator+(const Surreal<RealType>& z, const float& i) {
    return Surreal<RealType>(i+z.real,z.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator+(const Surreal<RealType>& z, const double& i) {
    return Surreal<RealType>(i+z.real,z.imag);
}

template <typename RealType>
DECL Surreal<RealType> Surreal<RealType>::operator-() const {
    return Surreal<RealType>(-real,-imag);
}

template <typename RealType>
DECL Surreal<RealType> operator-(const Surreal<RealType>& a, const Surreal<RealType>& b) {
    return Surreal<RealType>(a.real - b.real, a.imag - b.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator-(const Surreal<RealType>& a, const float& b) {
    return Surreal<RealType>(a.real - b, a.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator-(const Surreal<RealType>& a, const double& b) {
    return Surreal<RealType>(a.real - b, a.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator-(const Surreal<RealType>& a, const int& b) {
    return Surreal<RealType>(a.real - b, a.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator-(const float& b, const Surreal<RealType>& a) {
    return Surreal<RealType>(b - a.real, -a.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator-(const double& b, const Surreal<RealType>& a) {
    return Surreal<RealType>(b - a.real, -a.imag);
}

template <typename RealType>
DECL Surreal<RealType> operator-(const int& b, const Surreal<RealType>& a) {
    return Surreal<RealType>(b - a.real, -a.imag);
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
DECL Surreal<RealType> operator*(
    const Surreal<RealType>& a,
    const Surreal<RealType>& b) {
    return Surreal<RealType>(
        b.real*a.real, // we don't do b.imag*b.imag cuz autodiff
        b.real*a.imag + a.real*b.imag
    );
}

template <typename RealType>
DECL Surreal<RealType> operator*(const float& i, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        i*z.real,
        i*z.imag
    );
}

template <typename RealType>
DECL Surreal<RealType> operator*(const double& i, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        i*z.real,
        i*z.imag
    );
}

template <typename RealType>
DECL Surreal<RealType> operator*(const int& i, const Surreal<RealType>& z) {
    return Surreal<RealType>(
        i*z.real,
        i*z.imag
    );
}

template <typename RealType>
DECL Surreal<RealType> operator*(const Surreal<RealType>& z, const float& i) {
    return Surreal<RealType>(
        i*z.real,
        i*z.imag
    );
}

template <typename RealType>
DECL Surreal<RealType> operator*(const Surreal<RealType>& z, const double& i) {
    return Surreal<RealType>(
        i*z.real,
        i*z.imag
    );
}

template <typename RealType>
DECL Surreal<RealType> operator*(const Surreal<RealType>& z, const int& i) {
    return Surreal<RealType>(
        i*z.real,
        i*z.imag
    );
}

template <typename RealType>
DECL Surreal<RealType> operator/(
    const Surreal<RealType>& a,
    const Surreal<RealType>& b) {
    return Surreal<RealType>(
        a.real/b.real,
        (b.real*a.imag - a.real*b.imag)/(b.real*b.real)
    );
}

//

template <typename RealType>
DECL Surreal<RealType> operator/(
    const Surreal<RealType> & z,
    const int& r) {
    return Surreal<RealType>(z.real/r, z.imag/r);
}

template <typename RealType>
DECL Surreal<RealType> operator/(
    const Surreal<RealType> & z,
    const float& r) {
    return Surreal<RealType>(z.real/r, z.imag/r);
}

template <typename RealType>
DECL Surreal<RealType> operator/(
    const Surreal<RealType> & z,
    const double& r) {
    return Surreal<RealType>(z.real/r, z.imag/r);
}

// dividing a real by a complex is different from divding a complex by a real

template <typename RealType>
DECL Surreal<RealType> operator/(
    const int& r,
    const Surreal<RealType> & z) {
    return Surreal<RealType>(r/z.real, -r*z.imag/(z.real*z.real));
}

template <typename RealType>
DECL Surreal<RealType> operator/(
    const float& r,
    const Surreal<RealType> & z) {
    return Surreal<RealType>(r/z.real, -r*z.imag/(z.real*z.real));
}

template <typename RealType>
DECL Surreal<RealType> operator/(
    const double& r,
    const Surreal<RealType> & z) {
    return Surreal<RealType>(r/z.real, -r*z.imag/(z.real*z.real));
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
template <typename OtherType>
DECL Surreal<RealType>& Surreal<RealType>::operator/=(const OtherType& i) {
    real /= RealType(i);
    imag /= RealType(i); // wtf?
    return *this;
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
DECL Surreal<RealType> tanh(const Surreal<RealType>& z) {
    double coshv = cosh(z.real);
    return Surreal<RealType>(
        tanh(z.real),
        z.imag/(coshv*coshv)
    );
}

template <typename RealType>
DECL Surreal<RealType> cosh(const Surreal<RealType>& z) {
    double coshv = cosh(z.real);
    double sinhv = sinh(z.real);
    return Surreal<RealType>(
        coshv,
        z.imag*sinhv
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
DECL Surreal<RealType> log(const Surreal<RealType>& z) {
    return Surreal<RealType>(log(z.real), z.imag/z.real);
}


template <typename RealType>
DECL Surreal<RealType> sqrt(const Surreal<RealType>& z) {
    RealType sqrtv = sqrt(z.real);

    return Surreal<RealType>(
        sqrtv,
        0.5*z.imag/sqrtv
    );
}

template <typename RealType>
DECL Surreal<RealType> rsqrt(const Surreal<RealType>& z) {
    RealType rsqrta = 1/sqrt(z.real);

    return Surreal<RealType>(
        rsqrta,
        -(z.imag*rsqrta)/(2*z.real)
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

template <typename RealType>
DECL Surreal<RealType> floor(const Surreal<RealType>& z) {
    return Surreal<RealType>(floor(z.real),0.);
}
