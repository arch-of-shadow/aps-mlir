module {
  func.func @au_generated_candidate_1__direct(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<32x3xf32>, %arg4: memref<32x3xf32>, %arg5: memref<f32>, %arg6: memref<f32>, %arg7: memref<f32>, %arg8: memref<f32>) attributes {area_square_microns = 5197.9886999999999 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 335.22336000000001 : f64, cycle = 28129 : i64, datapath_area_square_microns = 4862.7653399999999 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 28129 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-build-aps-icp-32/target-resource.json", schedule = true} {
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c0_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : i32
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.000000e+30 : f32
    %c32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 32 : index
    scf.for %arg9 = %c0 to %c32 step %c1 {
      %alloca = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      memref.store %cst, %alloca[] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      %alloca_0 = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<i32>
      memref.store %c0_i32, %alloca_0[] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<i32>
      scf.for %arg10 = %c0 to %c32 step %c1 {
        %26 = arith.index_cast %arg10 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
        %27 = memref.load %arg0[] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<f32>
        %28 = memref.load %arg1[] {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<f32>
        %29 = memref.load %arg2[] {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<f32>
        %30 = memref.load %arg3[%arg9, %c0] {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<32x3xf32>
        %31 = arith.addf %30, %27 {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : f32
        %32 = memref.load %arg4[%arg10, %c0] {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : memref<32x3xf32>
        %33 = arith.subf %31, %32 {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : f32
        %34 = memref.load %arg3[%arg9, %c1] {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : memref<32x3xf32>
        %35 = arith.addf %34, %28 {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : f32
        %36 = memref.load %arg4[%arg10, %c1] {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : memref<32x3xf32>
        %37 = arith.subf %35, %36 {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : f32
        %38 = memref.load %arg3[%arg9, %c2] {endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : memref<32x3xf32>
        %39 = arith.addf %38, %29 {endtime = 13 : i64, ref_endtime = 13 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : f32
        %40 = memref.load %arg4[%arg10, %c2] {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : memref<32x3xf32>
        %41 = arith.subf %39, %40 {endtime = 16 : i64, ref_endtime = 16 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : f32
        %42 = arith.mulf %33, %33 {endtime = 15 : i64, ref_endtime = 15 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : f32
        %43 = arith.mulf %37, %37 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : f32
        %44 = arith.addf %42, %43 {endtime = 20 : i64, ref_endtime = 20 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : f32
        %45 = arith.mulf %41, %41 {endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : f32
        %46 = arith.addf %44, %45 {endtime = 23 : i64, ref_endtime = 23 : i64, ref_starttime = 20 : i64, starttime = 20 : i64} : f32
        %47 = memref.load %alloca[] {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : memref<f32>
        %48 = arith.cmpf olt, %46, %47 {endtime = 24 : i64, ref_endtime = 24 : i64, ref_starttime = 23 : i64, starttime = 23 : i64} : f32
        scf.if %48 {
          memref.store %46, %alloca[] {endtime = 25 : i64, ref_endtime = 25 : i64, ref_starttime = 24 : i64, starttime = 24 : i64} : memref<f32>
          memref.store %26, %alloca_0[] {endtime = 26 : i64, ref_endtime = 26 : i64, ref_starttime = 25 : i64, starttime = 25 : i64} : memref<i32>
        } {cycle = 2 : i64, endtime = 26 : i64, ref_endtime = 26 : i64, ref_starttime = 24 : i64, starttime = 24 : i64}
      } {cycle = 833 : i64, endtime = 833 : i64, ref_endtime = 833 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 32 : i64}
      %0 = memref.load %alloca_0[] {endtime = 834 : i64, ref_endtime = 834 : i64, ref_starttime = 833 : i64, starttime = 833 : i64} : memref<i32>
      %1 = arith.index_cast %0 {endtime = 834 : i64, ref_endtime = 834 : i64, ref_starttime = 834 : i64, starttime = 834 : i64} : i32 to index
      %2 = memref.load %arg4[%1, %c0] {endtime = 835 : i64, ref_endtime = 835 : i64, ref_starttime = 834 : i64, starttime = 834 : i64} : memref<32x3xf32>
      %3 = memref.load %arg3[%arg9, %c0] {endtime = 836 : i64, ref_endtime = 836 : i64, ref_starttime = 835 : i64, starttime = 835 : i64} : memref<32x3xf32>
      %4 = memref.load %arg0[] {endtime = 837 : i64, ref_endtime = 837 : i64, ref_starttime = 836 : i64, starttime = 836 : i64} : memref<f32>
      %5 = arith.addf %3, %4 {endtime = 840 : i64, ref_endtime = 840 : i64, ref_starttime = 837 : i64, starttime = 837 : i64} : f32
      %6 = arith.subf %2, %5 {endtime = 843 : i64, ref_endtime = 843 : i64, ref_starttime = 840 : i64, starttime = 840 : i64} : f32
      %7 = memref.load %arg5[] {endtime = 838 : i64, ref_endtime = 838 : i64, ref_starttime = 837 : i64, starttime = 837 : i64} : memref<f32>
      %8 = arith.addf %7, %6 {endtime = 846 : i64, ref_endtime = 846 : i64, ref_starttime = 843 : i64, starttime = 843 : i64} : f32
      memref.store %8, %arg5[] {endtime = 847 : i64, ref_endtime = 847 : i64, ref_starttime = 846 : i64, starttime = 846 : i64} : memref<f32>
      %9 = memref.load %arg4[%1, %c1] {endtime = 848 : i64, ref_endtime = 848 : i64, ref_starttime = 847 : i64, starttime = 847 : i64} : memref<32x3xf32>
      %10 = memref.load %arg3[%arg9, %c1] {endtime = 849 : i64, ref_endtime = 849 : i64, ref_starttime = 848 : i64, starttime = 848 : i64} : memref<32x3xf32>
      %11 = memref.load %arg1[] {endtime = 850 : i64, ref_endtime = 850 : i64, ref_starttime = 849 : i64, starttime = 849 : i64} : memref<f32>
      %12 = arith.addf %10, %11 {endtime = 853 : i64, ref_endtime = 853 : i64, ref_starttime = 850 : i64, starttime = 850 : i64} : f32
      %13 = arith.subf %9, %12 {endtime = 856 : i64, ref_endtime = 856 : i64, ref_starttime = 853 : i64, starttime = 853 : i64} : f32
      %14 = memref.load %arg6[] {endtime = 851 : i64, ref_endtime = 851 : i64, ref_starttime = 850 : i64, starttime = 850 : i64} : memref<f32>
      %15 = arith.addf %14, %13 {endtime = 859 : i64, ref_endtime = 859 : i64, ref_starttime = 856 : i64, starttime = 856 : i64} : f32
      memref.store %15, %arg6[] {endtime = 860 : i64, ref_endtime = 860 : i64, ref_starttime = 859 : i64, starttime = 859 : i64} : memref<f32>
      %16 = memref.load %arg4[%1, %c2] {endtime = 861 : i64, ref_endtime = 861 : i64, ref_starttime = 860 : i64, starttime = 860 : i64} : memref<32x3xf32>
      %17 = memref.load %arg3[%arg9, %c2] {endtime = 862 : i64, ref_endtime = 862 : i64, ref_starttime = 861 : i64, starttime = 861 : i64} : memref<32x3xf32>
      %18 = memref.load %arg2[] {endtime = 863 : i64, ref_endtime = 863 : i64, ref_starttime = 862 : i64, starttime = 862 : i64} : memref<f32>
      %19 = arith.addf %17, %18 {endtime = 866 : i64, ref_endtime = 866 : i64, ref_starttime = 863 : i64, starttime = 863 : i64} : f32
      %20 = arith.subf %16, %19 {endtime = 869 : i64, ref_endtime = 869 : i64, ref_starttime = 866 : i64, starttime = 866 : i64} : f32
      %21 = memref.load %arg7[] {endtime = 864 : i64, ref_endtime = 864 : i64, ref_starttime = 863 : i64, starttime = 863 : i64} : memref<f32>
      %22 = arith.addf %21, %20 {endtime = 872 : i64, ref_endtime = 872 : i64, ref_starttime = 869 : i64, starttime = 869 : i64} : f32
      memref.store %22, %arg7[] {endtime = 873 : i64, ref_endtime = 873 : i64, ref_starttime = 872 : i64, starttime = 872 : i64} : memref<f32>
      %23 = memref.load %alloca[] {endtime = 874 : i64, ref_endtime = 874 : i64, ref_starttime = 873 : i64, starttime = 873 : i64} : memref<f32>
      %24 = memref.load %arg8[] {endtime = 875 : i64, ref_endtime = 875 : i64, ref_starttime = 874 : i64, starttime = 874 : i64} : memref<f32>
      %25 = arith.addf %24, %23 {endtime = 878 : i64, ref_endtime = 878 : i64, ref_starttime = 875 : i64, starttime = 875 : i64} : f32
      memref.store %25, %arg8[] {endtime = 879 : i64, ref_endtime = 879 : i64, ref_starttime = 878 : i64, starttime = 878 : i64} : memref<f32>
    } {cycle = 28129 : i64, endtime = 28129 : i64, ref_endtime = 28129 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 32 : i64}
    return
  }
}

