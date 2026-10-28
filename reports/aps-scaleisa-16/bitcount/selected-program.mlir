"builtin.module"() ({
  "memref.global"() <{constant, initial_value = dense<[[0], [1], [-1], [-2147483648], [2147483647], [1431655765], [-1431655766], [16843009], [-244068920], [-1893513343], [787661114], [-863880461], [1785836716], [136392293], [-1477400546], [1145053655], [-494953584], [-2134960823], [519999234], [-1086453573], [1568506484], [-71500755], [-1711507994], [935063455], [-730109608], [1916461841], [284843210], [-1363552637], [1333350460], [-315045387], [-1946664018], [691518823], [-949536992], [1689694425], [46541458], [-1607097269], [1046814212], [-541812803], [2105807222], [464751407], [-1184693016], [1462927009], [-188614566], [-1833864685], [811658188], [-835689083], [1812979518], [237983991], [-1469132112], [1250839657], [-456276446], [-2027077669], [560773524], [-1018416307], [1569434886], [-5560641], [-1637179272], [940186161], [-624323606], [1957236131], [392726364], [-1320681195], [1394047694], [-316214137]]> : tensor<64x1xi32>, sym_name = "inputs", sym_visibility = "private", type = memref<64x1xi32>}> : () -> ()
  "memref.global"() <{initial_value, sym_name = "scalar_results", type = memref<64xi32>}> : () -> ()
  "func.func"() <{function_type = () -> i32, sym_name = "scalar_selected_main"}> ({
    %10 = "arith.constant"() <{value = 1 : index}> : () -> index
    %11 = "arith.constant"() <{value = 64 : index}> : () -> index
    %12 = "arith.constant"() <{value = 0 : index}> : () -> index
    %13 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %14 = "memref.get_global"() <{name = @inputs}> : () -> memref<64x1xi32>
    %15 = "memref.get_global"() <{name = @scalar_results}> : () -> memref<64xi32>
    "scf.for"(%12, %11, %10) ({
    ^bb0(%arg3: index):
      %16 = "memref.load"(%14, %arg3, %12) : (memref<64x1xi32>, index, index) -> i32
      %17 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
      "memref.store"(%13, %17) : (i32, memref<i32>) -> ()
      "memref.store"(%13, %17) : (i32, memref<i32>) -> ()
      %18 = "func.call"(%16) <{callee = @au_generated_candidate_9__l0u_site_22}> : (i32) -> i32
      "memref.store"(%18, %17) : (i32, memref<i32>) -> ()
      %19 = "memref.load"(%17) : (memref<i32>) -> i32
      "memref.store"(%19, %15, %arg3) : (i32, memref<64xi32>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"(%13) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32) -> i32, sym_name = "au_generated_candidate_9__l0u_site_22"}> ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 0 : index}> : () -> index
    %2 = "arith.constant"() <{value = 32 : index}> : () -> index
    %3 = "arith.constant"() <{value = 1 : index}> : () -> index
    %4 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %5 = "scf.for"(%1, %2, %3, %4) ({
    ^bb0(%arg1: index, %arg2: i32):
      %6 = "arith.index_cast"(%arg1) : (index) -> i32
      %7 = "arith.shrui"(%arg0, %6) : (i32, i32) -> i32
      %8 = "arith.andi"(%7, %0) : (i32, i32) -> i32
      %9 = "arith.addi"(%arg2, %8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "scf.yield"(%9) : (i32) -> ()
    }) : (index, index, index, i32) -> i32
    "func.return"(%5) : (i32) -> ()
  }) {au.generated = true} : () -> ()
}) : () -> ()
