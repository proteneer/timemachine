#pragma once

template<typename T>
int sign(T a) {
  return (a > 0) - (a < 0);
}