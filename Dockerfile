# copied from https://github.com/BraTS/Instructions/blob/master/docker_templates/template_2020/Dockerfile_CUDA
# for a GPU app use this Dockerfile, delete the Dockerfile_CPU then.
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# TODO
# fill in your info here
LABEL author="matheus.abrantesc@gmail.com"
LABEL application="cerqueira matheus"
LABEL maintainer="matheus.abrantesc@gmail.com"
# specify version here, if possible use semantic versioning
LABEL version="0.0.1"
LABEL status="beta"

# basic
RUN apt-get -y update && \
    apt-get install --no-install-recommends -y \
    apt-utils wget git tar build-essential curl nano vim

# install python 3.8
RUN apt-get install --no-install-recommends -y \
    python3.8 python3-pip python3.8-dev python3-dev 

# install all python requirements
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install setuptools wheel
# change according to your GPU driver version
RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# copy all files
COPY ./ ./
# RUN pip3 install -U python-dotenv

ENTRYPOINT [ "python3", "-u", "src/experiment.py" ]

