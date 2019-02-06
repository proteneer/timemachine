PLATFORM_FLAGS="";
if [ "$(uname)" == "Darwin" ]; then
	PLATFORM_FLAGS="-undefined dynamic_lookup";
    # Do something under Mac OS X platform        
fi

# g++ -g -O0 -march=native -Wall -shared -std=c++11 -fPIC $PLATFORM_FLAGS `python3 -m pybind11 --includes` wrappers.cpp -o energy`python3-config --extension-suffix`

g++ -O3 -ffast-math -march=native -Wall -shared -std=c++11 -fPIC $PLATFORM_FLAGS `python3 -m pybind11 --includes` wrappers.cpp -o energy`python3-config --extension-suffix`
