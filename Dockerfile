# Libraries required by RDkit
ARG LIBXRENDER_VERSION=1:0.9.10-*
ARG LIBXEXT_VERSION=2:1.3.4-*

FROM nvidia/cuda:11.7.1-devel-ubuntu20.04 AS tm_base_env
ARG LIBXRENDER_VERSION
ARG LIBXEXT_VERSION

# Copied out of anaconda's dockerfile
ARG MINICONDA_VERSION=py310_23.1.0-1
ARG MAKE_VERSION=4.2.1-1.2
ARG GIT_VERSION=1:2.25.1-*
ARG WGET_VERSION=1.20.3-1ubuntu2
RUN (apt-get update || true)  && apt-get install --no-install-recommends -y \
    wget=${WGET_VERSION} git=${GIT_VERSION} make=${MAKE_VERSION} libxrender1=${LIBXRENDER_VERSION} libxext-dev=${LIBXEXT_VERSION} vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Setup CMake
ARG CMAKE_VERSION=3.24.3
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
    conda clean -a && \
    conda activate ${ENV_NAME}

ENV PATH /opt/conda/envs/${ENV_NAME}/bin:$PATH

ENV CONDA_DEFAULT_ENV ${ENV_NAME}

# Install OpenMM
ARG OPENMM_VERSION=8.0.0

ARG DOXYGEN_VERSION=1.9.1
ARG CYTHON_VERSION=0.29.26
ARG SWIG_VERSION=3.0.12

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda install -y -c conda-forge swig=${SWIG_VERSION} doxygen=${DOXYGEN_VERSION} cython=${CYTHON_VERSION} && \
    conda clean -a

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
RUN pip install --no-cache-dir pre-commit==3.2.1
COPY .pre-commit-config.yaml /code/timemachine/
RUN cd /code/timemachine && git init . && pre-commit install-hooks

# Install CI requirements
COPY ci/requirements.txt /code/timemachine/ci/requirements.txt
RUN pip install --no-cache-dir -r timemachine/ci/requirements.txt

# Container that contains the cuda developer tools which allows building the customs ops
# Used as an intermediate for creating a final slimmed down container with timemachine and only the cuda runtime
FROM tm_base_env AS timemachine_cuda_dev
ARG CUDA_ARCH
ENV CMAKE_ARGS="-DCUDA_ARCH:STRING=${CUDA_ARCH}"

COPY . /code/timemachine/
WORKDIR /code/timemachine/
RUN pip install --no-cache-dir -e . && rm -rf ./build

# Container with only cuda runtime, half the size of the timemachine_cuda_dev container
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04 as timemachine
ARG LIBXRENDER_VERSION
ARG LIBXEXT_VERSION
RUN (apt-get update || true) && apt-get install --no-install-recommends -y libxrender1=${LIBXRENDER_VERSION} libxext-dev=${LIBXEXT_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --from=timemachine_cuda_dev /opt/ /opt/
COPY --from=timemachine_cuda_dev /code/ /code/
COPY --from=timemachine_cuda_dev /root/.bashrc /root/.bashrc
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
ARG ENV_NAME=timemachine
ENV PATH /opt/conda/envs/${ENV_NAME}/bin:$PATH
ENV CONDA_DEFAULT_ENV ${ENV_NAME}
