FROM uvxiao/aps-mlir:v0

WORKDIR /root

# Copy the APS source code
COPY . /root/aps

# Run pixi setup commands to build .pixi with correct container paths
# RUN /root/.pixi/bin/pixi run setup-ortools
# RUN /root/.pixi/bin/pixi run setup
# RUN /root/.pixi/bin/pixi run build
# RUN /root/.pixi/bin/pixi run fix_verilator

CMD [ "/bin/bash", "-l" ]
