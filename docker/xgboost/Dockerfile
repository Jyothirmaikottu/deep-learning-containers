# ============================================================================
# XGBoost DLC — Ubuntu 22.04 / Python 3.12 (CUDA 12.9)
# Multi-stage build:
#
#   sqlite-builder ─── (compiles SQLite from source)
#   wheel-builder ──── (builds sagemaker_xgboost_container wheel)
#   base ──────────── (system Python, uv, XGBoost, deps)
#   xgboost-sagemaker  (SageMaker serving layer on top of base)
#
# Based on: github.com/aws/sagemaker-xgboost-container/docker/3.0-5
# ============================================================================

# ── Global ARGs ─────────────────────────────────────────────────────────────
ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.9.1
ARG PYTHON_VERSION=3.12
ARG XGBOOST_VERSION=3.2.0
ARG SAGEMAKER_XGBOOST_VERSION=3.0-5

# ── Stage: sqlite-builder ───────────────────────────────────────────────────
FROM ubuntu:${UBUNTU_VERSION} AS sqlite-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential wget ca-certificates \
  && cd /tmp \
  && wget https://www.sqlite.org/2025/sqlite-autoconf-3500200.tar.gz \
  && tar xzf sqlite-autoconf-3500200.tar.gz \
  && cd sqlite-autoconf-3500200 \
  && ./configure --prefix=/usr/local && make && make install && ldconfig \
  && cd / && rm -rf /tmp/sqlite-autoconf-* \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Stage: wheel-builder ───────────────────────────────────────────────────
FROM ubuntu:${UBUNTU_VERSION} AS wheel-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 python3-pip python3-setuptools python3-wheel git ca-certificates \
  && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/aws/sagemaker-xgboost-container.git /build
WORKDIR /build
RUN python3 setup.py bdist_wheel --universal

# ── Stage: base ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS base

ARG PYTHON_VERSION
ARG XGBOOST_VERSION

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ENV DEBIAN_FRONTEND=noninteractive \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=utf-8

# System dependencies + Python 3.12
RUN apt-get update && apt-get -y upgrade \
  && apt-get -y install --no-install-recommends \
    build-essential curl git jq libatlas-base-dev expat nginx \
    openjdk-11-jdk-headless unzip wget \
    libxml2 libglib2.0-0 libffi-dev libsqlite3-0 \
    ca-certificates gnupg software-properties-common \
    libcurl4-openssl-dev libssl-dev zlib1g-dev zstd \
    python3 python3-dev python3-pip python3-venv \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create venv
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copy compiled SQLite from builder stage
COPY --from=sqlite-builder /usr/local/bin/sqlite3 /usr/local/bin/sqlite3
COPY --from=sqlite-builder /usr/local/lib/libsqlite3.* /usr/local/lib/
COPY --from=sqlite-builder /usr/local/include/sqlite3*.h /usr/local/include/
RUN ldconfig && echo "/usr/local/lib" >/etc/ld.so.conf.d/sqlite3.conf && ldconfig

RUN sqlite3 --version

# Install Python dependencies via uv
COPY docker/xgboost/pyproject.toml docker/xgboost/uv.lock /tmp/build/
RUN --mount=type=cache,target=/root/.cache/uv \
  cd /tmp/build && uv sync --frozen --no-dev --no-install-project \
  && rm -rf /tmp/build

ENV LD_LIBRARY_PATH=/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

# ── Stage: xgboost-sagemaker ───────────────────────────────────────────────
FROM base AS xgboost-sagemaker

ARG SAGEMAKER_XGBOOST_VERSION
ARG PYTHON_VERSION
ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE

# Telemetry
COPY ./scripts/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
COPY ./scripts/common/setup_oss_compliance.sh setup_oss_compliance.sh
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && bash setup_oss_compliance.sh python3 && rm setup_oss_compliance.sh \
  && rm -rf /root/oss_compliance* \
  && rm -rf /tmp/tmp* \
  && rm -rf /var/lib/apt/lists/*

# Install smdebug
RUN pip install --no-cache-dir git+https://github.com/awslabs/sagemaker-debugger.git@v1.0.32

# Patch sagemaker-inference decoder.py (CWE-502 fix)
COPY docker/xgboost/resources/patches/decoder.py /opt/venv/lib/python${PYTHON_VERSION}/site-packages/sagemaker_inference/decoder.py

# Patch collections.Mapping deprecation in sagemaker-containers
RUN sed -i 's/collections\.Mapping/collections.abc.Mapping/g' \
  /opt/venv/lib/python${PYTHON_VERSION}/site-packages/sagemaker_containers/_mapping.py

# Install sagemaker_xgboost_container wheel
COPY --from=wheel-builder /build/dist/sagemaker_xgboost_container-2.0-py2.py3-none-any.whl /tmp/
RUN pip install --force-reinstall PyYAML==6.0.1 \
  && pip install --no-cache --no-deps /tmp/sagemaker_xgboost_container-2.0-py2.py3-none-any.whl \
  && rm /tmp/sagemaker_xgboost_container-2.0-py2.py3-none-any.whl

# MMS setup
RUN useradd -m model-server \
  && mkdir -p /home/model-server/tmp && chown -R model-server /home/model-server
COPY docker/xgboost/resources/mms/config.properties.tmp /home/model-server
ENV XGBOOST_MMS_CONFIG=/home/model-server/config.properties
RUN mkdir -p /tmp/plugins
COPY docker/xgboost/resources/mms/endpoints-1.0.jar /tmp/plugins
RUN chmod +x /tmp/plugins/endpoints-1.0.jar

# Model directory
RUN mkdir -p /opt/ml/models && chmod +rwx /opt/ml/models

# Dask config
RUN mkdir -p /etc/dask
COPY docker/xgboost/resources/dask_configs.yaml /etc/dask/

# Labels
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# SageMaker environment
ENV SM_INPUT=/opt/ml/input \
  SM_INPUT_TRAINING_CONFIG_FILE=/opt/ml/input/config/hyperparameters.json \
  SM_INPUT_DATA_CONFIG_FILE=/opt/ml/input/config/inputdataconfig.json \
  SM_CHECKPOINT_CONFIG_FILE=/opt/ml/input/config/checkpointconfig.json \
  NCCL_SOCKET_IFNAME=eth \
  SM_MODEL_DIR=/opt/ml/model \
  SAGEMAKER_TRAINING_MODULE=sagemaker_xgboost_container.training:main \
  SAGEMAKER_SERVING_MODULE=sagemaker_xgboost_container.serving:main \
  TEMP=/home/model-server/tmp

EXPOSE 8080
