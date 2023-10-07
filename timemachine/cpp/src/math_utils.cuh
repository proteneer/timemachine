#pragma once

namespace timemachine {

int __forceinline__ ceil_divide(int x, int y) { return (x + y - 1) / y; };

} // namespace timemachine
