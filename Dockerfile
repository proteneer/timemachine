# Libraries required by RDkit
ARG LIBXRENDER_VERSION=1:0.9.10-*
ARG LIBXEXT_VERSION=2:1.3.4-*

FROM docker.io/nvidia/cuda:12.4.1-devel-ubuntu20.04 AS tm_base_env
ARG LIBXRENDER_VERSION
ARG LIBXEXT_VERSION

# Copied out of anaconda's dockerfile
ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=24.9.2-0
ARG MAKE_VERSION=4.2.1-1.2
ARG GIT_VERSION=1:2.25.1-*
ARG WGET_VERSION=1.20.3-*
RUN (apt-get update || true)  && apt-get install --no-install-recommends -y \
    wget=${WGET_VERSION} git=${GIT_VERSION} make=${MAKE_VERSION} libxrender1=${LIBXRENDER_VERSION} libxext-dev=${LIBXEXT_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O ~/miniconda.sh && \
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

WORKDIR /code/

# Copy the pip requirements to cache when possible
COPY requirements.txt /code/timemachine/
RUN pip install --no-cache-dir -r timemachine/requirements.txt

# NOTE: timemachine_ci must come before timemachine in the Dockerfile;
# otherwise, CI will try (and fail) to build timemachine to reach the
# timemachine_ci target
FROM tm_base_env AS timemachine_ci

# Install CI requirements
COPY ci/requirements.txt /code/timemachine/ci/requirements.txt
RUN pip install --no-cache-dir -r timemachine/ci/requirements.txt

# Install pre-commit and cache hooks
COPY .pre-commit-config.yaml /code/timemachine/
RUN cd /code/timemachine && git init . && pre-commit install-hooks


# Container that contains the cuda developer tools which allows building the customs ops
# Used as an intermediate for creating a final slimmed down container with timemachine and only the cuda runtime
FROM tm_base_env AS timemachine_cuda_dev
ARG CUDA_ARCH
ENV CMAKE_ARGS="-DCUDA_ARCH:STRING=${CUDA_ARCH}"

COPY . /code/timemachine/
WORKDIR /code/timemachine/
RUN pip install --no-cache-dir -e . && rm -rf ./build

# Container with only cuda base, half the size of the timemachine_cuda_dev container
# Need to copy curand/cudart as these are dependencies of the Timemachine GPU code
FROM docker.io/nvidia/cuda:12.4.1-base-ubuntu20.04 AS timemachine
ARG LIBXRENDER_VERSION
ARG LIBXEXT_VERSION
RUN (apt-get update || true) && apt-get install --no-install-recommends -y libxrender1=${LIBXRENDER_VERSION} libxext-dev=${LIBXEXT_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy curand libraries from image, only require cudart and curand
COPY --from=timemachine_cuda_dev /usr/local/cuda/targets/x86_64-linux/lib/libcurand* /usr/local/cuda/targets/x86_64-linux/lib/
COPY --from=timemachine_cuda_dev /usr/local/cuda/lib64/libcurand* /usr/local/cuda/lib64/

COPY --from=timemachine_cuda_dev /opt/conda/ /opt/conda/
COPY --from=timemachine_cuda_dev /code/ /code/
COPY --from=timemachine_cuda_dev /root/.bashrc /root/.bashrc
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
ARG ENV_NAME=timemachine
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
ENV PATH /opt/conda/envs/${ENV_NAME}/bin:$PATH
ENV CONDA_DEFAULT_ENV ${ENV_NAME}
