# Stage 1: Builder - compile .pixi with correct container paths
FROM uvxiao/aps-mlir:v0 AS builder

WORKDIR /root

# ARG GIT_REPO=https://github.com/arch-of-shadow/aps-mlir.git
# ARG GIT_BRANCH=aspdac

# # Recursively clone the repo with submodules 
# RUN git clone --branch "$GIT_BRANCH" "$GIT_REPO" /root/aps

COPY . /root/aps
RUN ls /root/aps
COPY ../tutorial_chipyard /root/aps-mlir-chipyard

WORKDIR /root/aps

ENV PATH="/root/.pixi/bin:${PATH}"

# Run pixi setup commands to build .pixi with correct container paths
RUN pixi run --help
RUN pixi run clean

RUN rm -rf circt/build
RUN pixi run setup-ortools
RUN pixi run setup-firtool
RUN pixi run setup-yosys-slang
RUN pixi run fix-verilator

# We assume outside has done running "init-submodules" provided by chipyard
RUN pixi run setup-chipyard-docker

# aps
RUN pixi run setup
RUN pixi run build

# # Stage 2: Runtime - copy .pixi from builder
# FROM uvxiao/aps-mlir:v0 AS runtime

# WORKDIR /root

# ARG GIT_REPO=https://github.com/arch-of-shadow/aps-mlir.git
# ARG GIT_BRANCH=aspdac

# # Shallow clone the repo (no submodules, no history)
# RUN git clone --branch "$GIT_BRANCH" "$GIT_REPO" /root/aps

# # Copy the pre-built .pixi from builder stage
# COPY --from=builder /root/aps/.pixi /root/aps/.pixi
# # Copy the aps/build and circt directories to avoid re-building
# COPY --from=builder /root/aps/build /root/aps/build
# COPY --from=builder /root/aps/circt /root/aps/circt

WORKDIR /root


CMD [ "/bin/bash", "-l" ]
