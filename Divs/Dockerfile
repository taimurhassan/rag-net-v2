FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev=3.8* \
    python3-pip \
    python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# add the source code
WORKDIR /
ADD . .
