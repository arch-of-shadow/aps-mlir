# Stage 1: Builder - compile .pixi with correct container paths
FROM uvxiao/aps-mlir:v0 AS builder

WORKDIR /root

# Copy the APS source code
COPY . /root/aps

# Run pixi setup commands to build .pixi with correct container paths
RUN /root/.pixi/bin/pixi run -manifest-path /root/aps/pixi.toml setup-ortools
RUN /root/.pixi/bin/pixi run -manifest-path /root/aps/pixi.toml setup
RUN /root/.pixi/bin/pixi run -manifest-path /root/aps/pixi.toml build
RUN /root/.pixi/bin/pixi run -manifest-path /root/aps/pixi.toml fix_verilator

# Stage 2: Runtime - copy .pixi from builder
FROM uvxiao/aps-mlir:v0 AS runtime

WORKDIR /root

ARG GIT_REPO=https://github.com/pku-liang/aps-mlir.git
ARG GIT_BRANCH=aspdac

# Shallow clone the repo (no submodules, no history)
RUN git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO" /root/aps

# Copy the pre-built .pixi from builder stage
COPY --from=builder /root/aps/.pixi /root/aps/.pixi

CMD [ "/bin/bash", "-l" ]
