# Stage 1: Builder - compile .pixi with correct container paths
FROM uvxiao/aps-mlir:v0 AS builder

WORKDIR /root

ARG GIT_REPO=https://github.com/arch-of-shadow/aps-mlir.git
ARG GIT_BRANCH=aspdac

# Recursively clone the repo with submodules 
RUN git clone --branch "$GIT_BRANCH" "$GIT_REPO" /root/aps

WORKDIR /root/aps

ENV PATH="/root/.pixi/bin:${PATH}"

# Run pixi setup commands to build .pixi with correct container paths
RUN pixi run --help
RUN pixi run setup-ortools
RUN pixi run setup
RUN pixi run build
RUN pixi run fix_verilator

# Stage 2: Runtime - copy .pixi from builder
FROM uvxiao/aps-mlir:v0 AS runtime

WORKDIR /root

ARG GIT_REPO=https://github.com/arch-of-shadow/aps-mlir.git
ARG GIT_BRANCH=aspdac

# Shallow clone the repo (no submodules, no history)
RUN git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO" /root/aps

# Copy the pre-built .pixi from builder stage
COPY --from=builder /root/aps/.pixi /root/aps/.pixi

CMD [ "/bin/bash", "-l" ]
