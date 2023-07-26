#pragma once
#include <stdexcept>

class InvalidHardware : public std::exception {
public:
    const char *what() const throw() { return "Either no GPU or the GPU is acting up"; }
};
