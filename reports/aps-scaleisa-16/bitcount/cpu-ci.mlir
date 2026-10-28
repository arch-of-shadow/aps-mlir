module attributes {scaleisa.ci_instructions = 1 : i64, scaleisa.ci_sites = 1 : i64} {
  memref.global "private" constant @inputs : memref<64x1xi32> = dense<[[0], [1], [-1], [-2147483648], [2147483647], [1431655765], [-1431655766], [16843009], [-244068920], [-1893513343], [787661114], [-863880461], [1785836716], [136392293], [-1477400546], [1145053655], [-494953584], [-2134960823], [519999234], [-1086453573], [1568506484], [-71500755], [-1711507994], [935063455], [-730109608], [1916461841], [284843210], [-1363552637], [1333350460], [-315045387], [-1946664018], [691518823], [-949536992], [1689694425], [46541458], [-1607097269], [1046814212], [-541812803], [2105807222], [464751407], [-1184693016], [1462927009], [-188614566], [-1833864685], [811658188], [-835689083], [1812979518], [237983991], [-1469132112], [1250839657], [-456276446], [-2027077669], [560773524], [-1018416307], [1569434886], [-5560641], [-1637179272], [940186161], [-624323606], [1957236131], [392726364], [-1320681195], [1394047694], [-316214137]]>
  memref.global @scalar_results : memref<64xi32> = uninitialized
  func.func @scalar_selected_main() -> i32 {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @inputs : memref<64x1xi32>
    %1 = memref.get_global @scalar_results : memref<64xi32>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      %2 = memref.load %0[%arg0, %c0] : memref<64x1xi32>
      %alloca = memref.alloca() : memref<i32>
      memref.store %c0_i32, %alloca[] : memref<i32>
      memref.store %c0_i32, %alloca[] : memref<i32>
      %3 = llvm.inline_asm has_side_effects ".insn r 43, 6, 0, $0, $1, x0", "=r,r,~{memory}" %2 : (i32) -> i32
      memref.store %3, %alloca[] : memref<i32>
      %4 = memref.load %alloca[] : memref<i32>
      memref.store %4, %1[%arg0] : memref<64xi32>
    }
    return %c0_i32 : i32
  }
}

