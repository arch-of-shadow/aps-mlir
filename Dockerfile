# syntax=docker/dockerfile:1
# Build from a clean GitHub checkout, independent of the local build context:
#   docker build --platform=linux/amd64 -t aps-mlir .
#   docker build --platform=linux/amd64 --build-arg APS_REF=<branch-or-commit> -t aps-mlir .
#   docker run --rm -it aps-mlir
# APS's Pixi environment currently supports linux-64 only.
FROM ubuntu:24.04

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Pixi supplies project dependencies. apt only bootstraps Git/Pixi and the
# system compiler required by setup-verilator's /usr/bin/g++ and /usr/bin/ar.
ARG DEBIAN_FRONTEND=noninteractive
RUN test "$(uname -m)" = x86_64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

ARG PIXI_VERSION=0.67.1
RUN curl --fail --location --retry 3 \
        "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz" \
    | tar -xz -C /usr/local/bin pixi

# Public submodules use both SSH and HTTPS URLs. Use HTTPS without SSH keys.
RUN git config --global url."https://github.com/".insteadOf git@github.com:

ARG APS_REPOSITORY=https://github.com/pku-liang/aps-mlir.git
ARG APS_REF=main
WORKDIR /opt/aps-mlir
RUN git init . \
    && git remote add origin "${APS_REPOSITORY}" \
    && git fetch --depth=1 origin "${APS_REF}" \
    && git checkout --detach FETCH_HEAD

# Install dependencies at their final prefix; never copy a host .pixi environment.
RUN pixi install --locked

# Include Chipyard and its extra RV32 tools in every image.
RUN pixi run --frozen setup-chipyard

# Project tasks own setup and submodule initialization.
RUN pixi run --frozen build

RUN pixi run --frozen build-hardfloat-ip

# Keep the default shell in the activated project environment. For noninteractive
# commands, use: docker run --rm aps-mlir pixi run build/tools/aps-opt/aps-opt --help
# No external entrypoint script is required.
RUN pixi shell-hook --shell bash > /etc/profile.d/aps-pixi.sh
CMD ["/bin/bash", "-l"]
