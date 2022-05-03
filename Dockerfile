FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 AS tm_base_env

# Copied out of anaconda's dockerfile
ARG MINICONDA_VERSION=4.6.14
ARG MAKE_VERSION=4.2.1-1.2
ARG GIT_VERSION=1:2.25.1-*
ARG WGET_VERSION=1.20.3-1ubuntu2
RUN apt-get update && apt-get install --no-install-recommends -y wget=${WGET_VERSION} git=${GIT_VERSION} make=${MAKE_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Setup CMake
ARG CMAKE_VERSION=3.22.1
RUN wget --quiet https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz -O cmake.tar.gz && \
    tar -xzf cmake.tar.gz && \
    rm -rf cmake.tar.gz

ENV PATH $PATH:$PWD/cmake-${CMAKE_VERSION}-linux-x86_64/bin/

# Copy the environment yaml to cache environment when possible
COPY environment.yml /code/timemachine/

WORKDIR /code/timemachine/

ARG ENV_NAME=timemachine

# Create Timemachine Env
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda env create -n "${ENV_NAME}" --force -f environment.yml && \
    conda activate ${ENV_NAME}
ENV PATH /opt/conda/envs/${ENV_NAME}/bin:$PATH

ENV CONDA_DEFAULT_ENV ${ENV_NAME}

# Install OpenMM
ARG OPENMM_VERSION=7.5.1

ARG DOXYGEN_VERSION=1.9.1
ARG CYTHON_VERSION=0.29.26
ARG SWIG_VERSION=3.0.12

RUN . /opt/conda/etc/profile.d/conda.sh && conda install -y -c conda-forge swig=${SWIG_VERSION} doxygen=${DOXYGEN_VERSION} cython=${CYTHON_VERSION}

WORKDIR /code/
RUN git clone https://github.com/openmm/openmm.git --branch "${OPENMM_VERSION}" && \
    cd openmm/ && \
    mkdir build && \
    cd build && \
    cmake \
      -DOPENMM_BUILD_CPU_LIB=OFF \
      -DOPENMM_BUILD_AMOEBA_CUDA_LIB=OFF \
      -DOPENMM_BUILD_AMOEBA_OPENCL_LIB=OFF \
      -DOPENMM_BUILD_CUDA_LIB=OFF \
      -DOPENMM_BUILD_CUDA_COMPILER_PLUGIN=OFF \
      -DOPENMM_BUILD_C_AND_FORTRAN_WRAPPERS=OFF \
      -DOPENMM_BUILD_DRUDE_CUDA_LIB=OFF \
      -DOPENMM_BUILD_DRUDE_OPENCL_LIB=OFF \
      -DOPENMM_BUILD_EXAMPLES=OFF \
      -DOPENMM_BUILD_OPENCL_LIB=OFF \
      -DOPENMM_BUILD_RPMD_CUDA_LIB=OFF \
      -DOPENMM_BUILD_RPMD_OPENCL_LIB=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/openmm_install \
      ../ && \
    make -j "$(nproc)" && \
    make -j "$(nproc)" install && \
    make PythonInstall && \
    cd /code/ && \
    rm -rf openmm/

# Copy the pip requirements to cache when possible
COPY requirements.txt /code/timemachine/
RUN pip install --no-cache-dir -r timemachine/requirements.txt

# NOTE: timemachine_ci must come before timemachine in the Dockerfile;
# otherwise, CI will try (and fail) to build timemachine to reach the
# timemachine_ci target
FROM tm_base_env AS timemachine_ci

# Install pre-commit and cache hooks
RUN pip install --no-cache-dir pre-commit==2.17.0
COPY .pre-commit-config.yaml /code/timemachine/
RUN cd /code/timemachine && git init . && pre-commit install-hooks

# Install CI requirements
COPY ci/requirements.txt /code/timemachine/ci/requirements.txt
RUN pip install --no-cache-dir -r timemachine/ci/requirements.txt

FROM tm_base_env AS timemachine
ARG CUDA_ARCH=75
ENV CMAKE_ARGS -DCUDA_ARCH=${CUDA_ARCH}
COPY . /code/timemachine/
WORKDIR /code/timemachine/
RUN CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip install --no-cache-dir -e .[dev,test]
