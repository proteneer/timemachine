FROM nvidia/cuda:11.2.0-devel-ubuntu18.04

RUN apt-get update \
    && apt-get install --no-install-recommends -y wget git make cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate timemachine" >> ~/.bashrc
# Ensures that things are faster to build a new docker container
ENV PATH /opt/conda/envs/timemachine/bin:$PATH
RUN . /opt/conda/etc/profile.d/conda.sh && conda create -c openeye -c conda-forge -n timemachine openeye-toolkits python=3.7 openmm rdkit==2021.03.1 && conda activate timemachine

COPY . /code/
ENV PYTHONPATH /code/:$PYTHONPATH

ENV CONDA_DEFAULT_ENV timemachine
ARG cuda_arch=sm_75
WORKDIR /code/
RUN make CUDA_ARCH=$cuda_arch build
RUN pip install --no-cache-dir -r requirements.txt
