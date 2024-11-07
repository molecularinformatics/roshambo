ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    build-essential \
    swig \
    git \
    cmake \
    wget \
    libboost-all-dev \
    libcairo2-dev \
    python3-all \
    python3-dev \
    python3-numpy \
    python3-cairo \
    python3-pip \
    python3-setuptools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV RDKIT_VERSION Release_2024_09_2
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------------

# Use an official base image
FROM base  AS rdkit-build-env

USER root

# ENV LANG C.UTF-8
WORKDIR /root

# compile and install rdkit from source code
RUN wget https://github.com/rdkit/rdkit/archive/$RDKIT_VERSION.tar.gz && \
    tar xfz $RDKIT_VERSION.tar.gz && \
    rm $RDKIT_VERSION.tar.gz && \
    cd /root/rdkit-$RDKIT_VERSION/External/INCHI-API && \
    ./download-inchi.sh && \
    cd /root/rdkit-$RDKIT_VERSION && \
    mkdir build && \
    cd build && \
    cmake \
        -D Py_ENABLE_SHARED=1 \
        -D PYTHON_EXECUTABLE=/usr/bin/python3 \
        -D RDK_INSTALL_INTREE=ON \
        -D RDK_BUILD_INCHI_SUPPORT=ON \
        -D RDK_BUILD_AVALON_SUPPORT=ON \
        -D RDK_BUILD_PYTHON_WRAPPERS=ON \
        -D RDK_BUILD_CAIRO_SUPPORT=ON \
        -D RDK_INSTALL_STATIC_LIBS=OFF \
        -D RDK_BUILD_CPP_TESTS=OFF \
        -D RDK_BUILD_YAEHMOP_SUPPORT=ON \
        -D RDK_BUILD_XYZ2MOL_SUPPORT=ON \
        -D RDK_BUILD_FREESASA_SUPPORT=ON \
        -D CMAKE_VERBOSE_MAKEFILE:BOOL=ON \
        -D PYTHON_NUMPY_INCLUDE_PATH="$(python -c 'import numpy ; print(numpy.get_include())')" \
        .. && \
    make -j `grep -c ^processor /proc/cpuinfo` && \
    make install 

# -----------------------------------------------------------------------------------

FROM base AS roshambo-env

ARG USER_ID
ARG GROUP_ID

# Create a group and user with the same UID and GID as the host user
RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -m -u ${USER_ID} -g appuser -s /bin/bash appuser

# Copy rdkit installation from rdkit-build-env
COPY --from=rdkit-build-env /root/rdkit-$RDKIT_VERSION /home/appuser/rdkit-$RDKIT_VERSION

RUN chown -R appuser:appuser /home/appuser/rdkit-$RDKIT_VERSION

ENV RDBASE /home/appuser/rdkit-$RDKIT_VERSION
ENV RDKIT_LIB_DIR $RDBASE/lib
ENV LD_LIBRARY_PATH $RDBASE/lib
ENV RDKIT_INCLUDE_DIR $RDBASE/Code
ENV RDKIT_DATA_DIR $RDBASE/Data
ENV PYTHONPATH $PYTHONPATH:$RDBASE

ENV NVIDIA_VISIBLE_DEVICES=all

WORKDIR /home/appuser

USER appuser
# Install roshambo
RUN git clone https://github.com/rashatwi/roshambo.git \
# exporting CUDA_HOME from the CUDA_VERSION to keep only the first 2 element so 12.5.0 -> 12.5
 && export CUDA_HOME=/usr/local/cuda-$(echo ${CUDA_VERSION} | cut -d. -f1,2) \
 && cd roshambo \
 && pip install ipython \
 && pip install .

ENV PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /home/appuser/roshambo

ENTRYPOINT ["roshambo"]
CMD ["-h"]