#pragma once

#include "integrator.hpp"

namespace timemachine {

class Thermostat : public Integrator {

public:
    virtual double get_temperature() = 0;
};

} // end namespace timemachine
