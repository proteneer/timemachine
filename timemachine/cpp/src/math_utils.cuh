#pragma once

#include "surreal.cuh"

int __forceinline__ ceil_divide(int x, int y) { return (x + y - 1) / y; }

double __device__ __host__ __forceinline__ sign(double a) { return (a > 0) - (a < 0); }

float __device__ __host__ __forceinline__ sign(float a) { return (a > 0) - (a < 0); }

double __device__ __host__ __forceinline__ sign(Surreal<double> a) { return (a.real > 0) - (a.real < 0); }

float __device__ __host__ __forceinline__ sign(Surreal<float> a) { return (a.real > 0) - (a.real < 0); }
