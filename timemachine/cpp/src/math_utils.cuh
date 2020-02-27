#pragma once

__device__ __host__ double sign(double a) {
  return (a > 0) - (a < 0);
}

__device__ __host__ float sign(float a) {
  return (a > 0) - (a < 0);
}

__device__ __host__ double sign(Surreal<double> a) {
  return (a.real > 0) - (a.real < 0);
}

__device__ __host__ float sign(Surreal<float> a) {
  return (a.real > 0) - (a.real < 0);
}