FROM nvidia/cuda:11.2.0-devel-ubuntu18.04

# Copied out of anaconda's dockerfile
RUN apt-get update && apt-get install -y wget git make cmake
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate timemachine" >> ~/.bashrc
# Ensures that things are faster to build a new docker container
ENV PATH /opt/conda/envs/timemachine/bin:$PATH
RUN . /opt/conda/etc/profile.d/conda.sh && conda create -c openeye -c conda-forge -n timemachine openeye-toolkits python=3.7 openmm rdkit==2021.03.1 && conda activate timemachine

ENV PYTHONPATH /code/timemachine/:$PYTHONPATH
ADD timemachine/ /code/timemachine/

ENV CONDA_DEFAULT_ENV timemachine
ARG cuda_arch=sm_75
WORKDIR /code/timemachine/
RUN make CUDA_ARCH=$cuda_arch build
RUN pip install -r requirements.txt

ADD . /code/
WORKDIR /code/
RUN pip install -e . && pip install -r requirements.txt
