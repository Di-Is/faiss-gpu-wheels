##################################################
# Dockerfike for faiss-gpu
# Copyright (c) 2024 Di-Is
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
##################################################

ARG AUDITWHEEL_POLICY="manylinux2014_x86_64"

FROM quay.io/pypa/manylinux2014_x86_64 as manylinux2014_x86_64

ENV CUDA_REPO_URL=http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo


FROM quay.io/pypa/manylinux_2_28_x86_64 as manylinux_2_28_x86_64

ENV CUDA_REPO_URL=http://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Use GCC version 11 for compatibility CUDA 11.8
# Ref. https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version
RUN yum -y install \
       gcc-toolset-11-gcc \
       gcc-toolset-11-gcc-c++ \
       gcc-toolset-11-gcc-gfortran \
       gcc-toolset-11-binutils \
 && rm -rf /var/cache/yum/* \
 && yum clean all
ENV PATH="/opt/rh/gcc-toolset-11/root/usr/bin:${PATH}"


FROM ${AUDITWHEEL_POLICY}

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib64

RUN yum install -y openblas-devel openblas-static \
 && rm -rf /var/cache/yum/* \
 && yum clean all

ARG CUDA_MAJOR_VERSION
ARG CUDA_MINOR_VERSION
ARG CUDA_PATCH_VERSION

RUN yum -y install yum-utils \
 && yum-config-manager --add-repo ${CUDA_REPO_URL} \
 && yum -y install cuda-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION}-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}.${CUDA_PATCH_VERSION} \
 && ln -s cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} /usr/local/cuda \
 && rm -rf /var/cache/yum/* \
 && yum clean all
ENV PATH="/usr/local/cuda/bin:${PATH}"

COPY ./faiss /tmp/project/faiss
WORKDIR /tmp/project

ARG FAISS_OPT_LEVEL="generic"
ARG BUILD_NJOB=32
ARG CUDA_ARCHITECTURES="60-real;70-real;80-real;90-real"
RUN cd faiss \
 && cmake . \
       -B build \
       -DFAISS_ENABLE_GPU="ON" \
       -DFAISS_ENABLE_RAFT="OFF" \
       -DFAISS_ENABLE_PYTHON=OFF \
       -DBUILD_SHARED_LIBS=ON \
       -DBUILD_TESTING=OFF \
       -DFAISS_OPT_LEVEL=${FAISS_OPT_LEVEL} \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
  && make -C build -j${BUILD_NJOB} \
  && make -C build install
