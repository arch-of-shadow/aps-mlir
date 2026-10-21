//===- FloatIPLibrary.h - Standalone HardFloat extern bindings
//--------------===//
#ifndef APS_FLOATIPLIBRARY_H
#define APS_FLOATIPLIBRARY_H
#include "circt/Dialect/Cmt2/ECMT2/Circuit.h"
#include "circt/Dialect/Cmt2/ECMT2/Module.h"
namespace mlir {
circt::cmt2::ecmt2::ExternalModule *
createFloatIPModule(circt::cmt2::ecmt2::Circuit &circuit, unsigned operation,
                    unsigned latency, unsigned width = 32, unsigned predicate = 0);
} // namespace mlir
#endif
