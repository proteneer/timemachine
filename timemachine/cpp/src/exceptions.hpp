#pragma once
#include "cuda_runtime.h"
#include <stdexcept>
#include <string>

class InvalidHardware : public std::exception {
private:
    std::string err_msg_;

public:
    InvalidHardware(cudaError_t code) {
        err_msg_ = "Invalid Hardware - Code " + std::to_string(code) + ": " + cudaGetErrorString(code);
    }
    const char *what() const noexcept { return err_msg_.c_str(); }
};
