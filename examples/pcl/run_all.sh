pixi run opt examples/pcl/v3ddist_vs.cadl v3ddist_vs.mlir && pixi run sv v3ddist_vs.mlir v3ddist_vs.sv
pixi run opt examples/pcl/v3ddist_vv.cadl v3ddist_vv.mlir && pixi run sv v3ddist_vv.mlir v3ddist_vv.sv
pixi run opt examples/pcl/vcovmat3d.cadl vcovmat3d.mlir && pixi run sv vcovmat3d.mlir vcovmat3d.sv
pixi run opt examples/pcl/vcovmat3d_vs.cadl vcovmat3d_vs.mlir && pixi run sv vcovmat3d_vs.mlir vcovmat3d_vs.sv
pixi run opt examples/pcl/vfpsmax.cadl vfpsmax.mlir && pixi run sv vfpsmax.mlir vfpsmax.sv
pixi run opt examples/pcl/vgemv3d.cadl vgemv3d.mlir && pixi run sv vgemv3d.mlir vgemv3d.sv

# DECA
pixi run opt examples/deca/deca_decompress.cadl deca.mlir && pixi run sv deca.mlir deca.sv
pixi run opt examples/deca/gemm.cadl gemm.mlir && pixi run sv gemm.mlir gemm.sv
# PCL
pixi run opt examples/pcl/v3ddist_vv.cadl v3ddist_vv.mlir && pixi run sv v3ddist_vv.mlir v3ddist_vv.sv
pixi run opt examples/pcl/vcovmat3d_vs.cadl vcovmat3d_vs.mlir && pixi run sv vcovmat3d_vs.mlir vcovmat3d_vs.sv
pixi run opt examples/pcl/vfpsmax.cadl vfpsmax.mlir && pixi run sv vfpsmax.mlir vfpsmax.sv
pixi run opt examples/pcl/vgemv3d.cadl vgemv3d.mlir && pixi run sv vgemv3d.mlir vgemv3d.sv

# e2e
pixi run opt examples/pcl/pcl_all.cadl pcl.mlir && pixi run sv pcl.mlir pcl.sv
# pixi run opt examples/deca/deca_all.cadl deca_all.mlir && pixi run sv deca_all.mlir deca_all.sv
pixi run opt examples/deca/deca_all_i16.cadl deca_all_i16.mlir && pixi run sv deca_all_i16.mlir deca_all_i16.sv