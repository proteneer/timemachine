#pragma once

#include "energy.hpp"
#include "integrator.hpp" 
#include <vector>

namespace timemachine {

template <typename NumericType>
class Context {

private:

	std::vector<EnergyGPU<NumericType>*> energies_;
	Integrator<NumericType>* integrator_;

public:

	Context(
		std::vector<EnergyGPU<NumericType>* > energies,
		Integrator<NumericType>* intg
	) :  energies_(energies), integrator_(intg) {

	};

	void step();

};

}