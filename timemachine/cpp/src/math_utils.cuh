#pragma once

template<typename T>
__device__ __host__ int sign(T a) {
  return (a > 0) - (a < 0);
}