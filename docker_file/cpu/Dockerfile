FROM ubuntu:16.04

RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install build-essential
RUN apt-get -y install git

RUN apt-get -y install python-dev
RUN apt-get -y install python-pip

RUN pip install numpy
RUN pip install pandas

RUN apt-get -y install gfortran libatlas-base-dev

RUN pip install --upgrade pip
RUN pip install scipy
RUN pip install sklearn
RUN pip install keras
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl

RUN pip install tensorflow


RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN apt-get -y install vim
RUN pip install h5py
