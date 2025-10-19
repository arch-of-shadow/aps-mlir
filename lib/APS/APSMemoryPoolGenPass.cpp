//===- APSMemoryPoolGenPass.cpp - Generate scratchpad memory pool --------===//
//
// This pass generates a hierarchical CMT2-based scratchpad memory pool from
// aps.memorymap with burst access support.
//
// Architecture:
//   ScratchpadMemoryPool (top-level)
//     ├─ burst_read(addr: u64) -> u64
//     ├─ burst_write(addr: u64, data: u64)
//     ├─ memory_mem_a (submodule per memory entry)
//     │    ├─ burst_read(addr: u64) -> u64
//     │    ├─ burst_write(addr: u64, data: u64)
//     │    ├─ mem_a_bank0 (Mem1r1w instance)
//     │    └─ mem_a_bank1, ...
//     └─ memory_mem_b (submodule)
//          └─ ...
//
// Features:
//   - Hierarchical module structure with proper instantiation
//   - Address decoding for routing burst accesses to correct memory entry
//   - Bank selection logic (cyclic vs block partitioning)
//   - Data bit extraction/placement for 64-bit burst bus
//   - Support for u8/u16/u24/u32/u40/u48/u56/u64 data widths
//   - Bank conflict validation (requires num_banks × data_width >= 64)
//   - cmt2.call operations for method invocation
//
//===----------------------------------------------------------------------===//

#include "APS/Passes.h"
#include "APS/APSOps.h"
#include "circt/Dialect/Cmt2/ECMT2/Circuit.h"
#include "circt/Dialect/Cmt2/ECMT2/FunctionLike.h"
#include "circt/Dialect/Cmt2/ECMT2/Instance.h"
#include "circt/Dialect/Cmt2/ECMT2/Module.h"
#include "circt/Dialect/Cmt2/ECMT2/ModuleLibrary.h"
#include "circt/Dialect/Cmt2/Cmt2Dialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/APInt.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "aps-memory-pool-gen"

namespace mlir {

using namespace circt::cmt2::ecmt2;
using namespace circt::firrtl;

struct APSMemoryPoolGenPass : public PassWrapper<APSMemoryPoolGenPass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(APSMemoryPoolGenPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<aps::APSDialect>();
    registry.insert<circt::cmt2::Cmt2Dialect>();
    registry.insert<FIRRTLDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Find the aps.memorymap operation
    aps::MemoryMapOp memoryMapOp;
    moduleOp.walk([&](aps::MemoryMapOp op) {
      memoryMapOp = op;
      return WalkResult::interrupt();
    });

    if (!memoryMapOp) {
      // No memory map found, nothing to do
      return;
    }

    // Initialize CMT2 module library
    auto &library = ModuleLibrary::getInstance();

    // Try to find the module library manifest
    // First try: relative to build directory
    llvm::SmallString<256> manifestPath;
    llvm::sys::path::append(manifestPath, "circt/lib/Dialect/Cmt2/ModuleLibrary/manifest.yaml");

    // Set library base path (parent of manifest)
    if (llvm::sys::fs::exists(manifestPath)) {
      llvm::SmallString<256> libraryPath = llvm::sys::path::parent_path(manifestPath);
      library.setLibraryPath(libraryPath);

      // Load the manifest
      if (mlir::failed(library.loadManifest(manifestPath))) {
        moduleOp.emitWarning() << "Failed to load module library manifest from "
                               << manifestPath;
        return;
      }
    } else {
      moduleOp.emitWarning() << "Module library manifest not found. "
                             << "External FIRRTL modules may not work correctly.";
      return;
    }

    // Generate the memory pool
    generateMemoryPool(moduleOp, memoryMapOp);
  }

  StringRef getArgument() const final { return "aps-memory-pool-gen"; }
  StringRef getDescription() const final {
    return "Generate scratchpad memory pool from aps.memorymap";
  }

private:
  /// Get memref.global by symbol name
  memref::GlobalOp getGlobalMemRef(ModuleOp moduleOp, StringRef symbolName) {
    auto globalOp = moduleOp.lookupSymbol<memref::GlobalOp>(symbolName);
    return globalOp;
  }

  /// Extract data width and address width from a memref type
  bool extractMemoryParameters(memref::GlobalOp globalOp,
                               int &dataWidth, int &addrWidth, int &depth) {
    if (!globalOp) {
      return false;
    }

    auto memrefType = globalOp.getType();
    if (!memrefType) {
      return false;
    }

    // Get element type and extract bit width
    auto elementType = memrefType.getElementType();
    if (auto intType = llvm::dyn_cast<IntegerType>(elementType)) {
      dataWidth = intType.getWidth();
    } else if (auto floatType = llvm::dyn_cast<FloatType>(elementType)) {
      dataWidth = floatType.getWidth();
    } else {
      return false;
    }

    // Get memory depth (number of elements)
    if (memrefType.getRank() != 1) {
      // Only support 1D memrefs for now
      return false;
    }

    depth = memrefType.getShape()[0];

    // Calculate address width: ceil(log2(depth))
    addrWidth = depth == 0 ? 1 : llvm::Log2_32_Ceil(depth);

    return true;
  }

  /// Helper struct for memory entry information
  struct MemoryEntryInfo {
    std::string name;
    uint32_t baseAddress;
    uint32_t bankSize;
    uint32_t numBanks;
    bool isCyclic;
    int dataWidth;
    int addrWidth;
    int depth;
    llvm::SmallVector<Instance*, 4> bankInstances;
  };


  /// Generate a wire_default module that wraps a wire with default value
  /// This is similar to the Rust implementation in impl_backend.rs
  Module* generateWireDefaultModule(Circuit &circuit,
                                     uint32_t width,
                                     uint64_t defaultValue,
                                     const std::string &name) {
    auto *wireMod = circuit.addModule(name);
    auto &builder = wireMod->getBuilder();
    auto loc = wireMod->getLoc();

    auto wireType = UIntType::get(builder.getContext(), width);

    // Create base wire instance from library
    llvm::StringMap<int64_t> wireParams;
    wireParams["width"] = width;

    auto *baseWire = circuit.addExternalModule("Wire", wireParams);
    if (!baseWire) {
      llvm::errs() << "ERROR: Failed to create Wire external module!\n";
      return nullptr;
    }

    // Bind base wire methods according to manifest
    baseWire->bindMethod("write", "write_enable", "", {"write_data"}, {});
    baseWire->bindValue("read", "", {"read_data"});

    // Add base wire instance
    auto *innerWire = wireMod->addInstance("inner", baseWire, {});

    // Add read value that delegates to inner wire
    auto *readVal = wireMod->addValue("read", {wireType});
    readVal->guard([&](mlir::OpBuilder &b) {
      auto u1Type = UIntType::get(b.getContext(), 1);
      auto trueConst = b.create<ConstantOp>(loc, u1Type, llvm::APInt(1, 1));
      b.create<circt::cmt2::ReturnOp>(loc, trueConst.getResult());
    });
    readVal->body([&, innerWire](mlir::OpBuilder &b) {
      auto vals = innerWire->callValue("read", b);
      b.create<circt::cmt2::ReturnOp>(b.getUnknownLoc(), mlir::ValueRange{vals[0]});
    });
    readVal->finalize();

    // Add write method that delegates to inner wire
    auto *writeMethod = wireMod->addMethod("write", {{"in_", wireType}}, {});
    writeMethod->guard([&](mlir::OpBuilder &b, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto u1Type = UIntType::get(b.getContext(), 1);
      auto trueConst = b.create<ConstantOp>(loc, u1Type, llvm::APInt(1, 1));
      b.create<circt::cmt2::ReturnOp>(loc, trueConst.getResult());
    });
    writeMethod->body([&](mlir::OpBuilder &b, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto inVal = args[0];
      innerWire->callMethod("write", {inVal}, b);
      b.create<circt::cmt2::ReturnOp>(loc);
    });
    writeMethod->finalize();

    // Add default rule that writes default value
    auto *defaultRule = wireMod->addRule("default");
    defaultRule->guard([&](mlir::OpBuilder &b) {
      auto u1Type = UIntType::get(b.getContext(), 1);
      auto trueConst = b.create<ConstantOp>(loc, u1Type, llvm::APInt(1, 1));
      b.create<circt::cmt2::ReturnOp>(loc, trueConst.getResult());
    });
    defaultRule->body([&](mlir::OpBuilder &b) {
      auto defConst = b.create<ConstantOp>(loc, wireType, llvm::APInt(width, defaultValue));
      innerWire->callMethod("write", {defConst.getResult()}, b);
      b.create<circt::cmt2::ReturnOp>(loc);
    });
    defaultRule->finalize();

    // Set scheduling: write, default, read
    wireMod->setPrecedence({{"write", "default"}, {"default", "read"}});

    return wireMod;
  }

  /// Generate a bank wrapper module that encapsulates bank selection and data alignment
  Module* generateBankWrapperModule(const MemoryEntryInfo &entryInfo,
                                    Circuit &circuit,
                                    size_t bankIdx,
                                    ExternalModule *memMod,
                                    Clock clk, Reset rst) {
    std::string wrapperName = "BankWrapper_" + entryInfo.name + "_" + std::to_string(bankIdx);
    auto *wrapper = circuit.addModule(wrapperName);

    // Add clock and reset
    Clock wrapperClk = wrapper->addClockArgument("clk");
    Reset wrapperRst = wrapper->addResetArgument("rst");

    auto &builder = wrapper->getBuilder();
    auto loc = wrapper->getLoc();
    auto u64Type = UIntType::get(builder.getContext(), 64);

    uint32_t elementsPerBurst = 64 / entryInfo.dataWidth;

    // Create the actual memory bank instance inside this wrapper
    auto *bankInst = wrapper->addInstance(
      "mem_bank",
      memMod,
      {wrapperClk.getValue(), wrapperRst.getValue()}
    );

    // Create wire_default modules for enable, data, and addr
    // Save insertion point before creating wire modules
    auto savedIPForWires = wrapper->getBuilder().saveInsertionPoint();

    auto *wireEnableMod = generateWireDefaultModule(circuit, 1, 0,
                                                     "WireDefault_enable_" + std::to_string(bankIdx));
    auto *wireDataMod = generateWireDefaultModule(circuit, entryInfo.dataWidth, 0,
                                                   "WireDefault_data_" + std::to_string(bankIdx));
    auto *wireAddrMod = generateWireDefaultModule(circuit, entryInfo.addrWidth, 0,
                                                   "WireDefault_addr_" + std::to_string(bankIdx));

    // Restore insertion point back to wrapper
    wrapper->getBuilder().restoreInsertionPoint(savedIPForWires);

    // Create wire_default instances in the wrapper
    auto *writeEnableWire = wrapper->addInstance("write_enable_wire", wireEnableMod, {});
    auto *writeDataWire = wrapper->addInstance("write_data_wire", wireDataMod, {});
    auto *writeAddrWire = wrapper->addInstance("write_addr_wire", wireAddrMod, {});

    // burst_read method: returns 64-bit aligned data if address is for this bank, else 0
    auto *burstRead = wrapper->addMethod("burst_read", {{"addr", u64Type}}, {u64Type});

    burstRead->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto u1Type = UIntType::get(guardBuilder.getContext(), 1);
      auto trueConst = guardBuilder.create<ConstantOp>(loc, u1Type, llvm::APInt(1, 1));
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getResult());
    });

    burstRead->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = args[0];

      // Helper: Create constants
      auto burstSizeConst = bodyBuilder.create<ConstantOp>(loc, u64Type, llvm::APInt(64, 8));  // 64-bit burst = 8 bytes
      auto numBanksConst = bodyBuilder.create<ConstantOp>(loc, u64Type, llvm::APInt(64, entryInfo.numBanks));
      auto myBankConst = bodyBuilder.create<ConstantOp>(loc, u64Type, llvm::APInt(64, bankIdx));
      auto elementsPerBurstConst = bodyBuilder.create<ConstantOp>(loc, u64Type, llvm::APInt(64, elementsPerBurst));

      // Helper 1: burst_idx = addr / 8
      auto burstIdx = bodyBuilder.create<DivPrimOp>(loc, addr, burstSizeConst.getResult());

      // Helper 2: burst_pattern = burst_idx % num_banks
      auto burstPattern = bodyBuilder.create<RemPrimOp>(loc, burstIdx, numBanksConst.getResult());

      // Helper 3: Check if this bank participates: isMine = (burst_pattern <= my_bank) && (my_bank < burst_pattern + elements_per_burst)
      // Simplified for cyclic: position = (my_bank - burst_pattern + num_banks) % num_banks
      auto posTemp = bodyBuilder.create<SubPrimOp>(loc, myBankConst.getResult(), burstPattern);
      auto positionInSeq = bodyBuilder.create<AddPrimOp>(loc, posTemp, numBanksConst.getResult());
      auto position = bodyBuilder.create<RemPrimOp>(loc, positionInSeq, numBanksConst.getResult());
      auto isMine = bodyBuilder.create<LTPrimOp>(loc, position, elementsPerBurstConst.getResult());

      // Helper 4: Calculate local offset
      // For banks in pattern: offset = (burst_idx - pattern_offset) / num_banks
      // pattern_offset = my_bank - burst_pattern (when my_bank >= burst_pattern) or my_bank + num_banks - burst_pattern (wrapped)
      // Simplified: offset = burst_idx / num_banks (for pattern 0), adjust for other patterns
      auto patternOffset = bodyBuilder.create<SubPrimOp>(loc, myBankConst.getResult(), burstPattern);
      auto patternOffsetAdjusted = bodyBuilder.create<AddPrimOp>(loc, patternOffset, numBanksConst.getResult());
      auto patternOffsetMod = bodyBuilder.create<RemPrimOp>(loc, patternOffsetAdjusted, numBanksConst.getResult());

      // adjusted_burst_idx = burst_idx - pattern_offset_mod
      auto adjustedBurstIdx = bodyBuilder.create<SubPrimOp>(loc, burstIdx, patternOffsetMod);
      auto localOffset = bodyBuilder.create<DivPrimOp>(loc, adjustedBurstIdx, numBanksConst.getResult());

      // Truncate to address width
      auto bankAddrType = UIntType::get(bodyBuilder.getContext(), entryInfo.addrWidth);
      auto localAddrTrunc = bodyBuilder.create<BitsPrimOp>(
        loc, bankAddrType, localOffset, entryInfo.addrWidth - 1, 0);

      // Read from bank
      bankInst->callMethod("rd0", {localAddrTrunc}, bodyBuilder);
      auto bankDataValues = bankInst->callValue("rd1", bodyBuilder);
      auto rawData = bankDataValues[0];

      // Helper 5: Calculate bit position: position * data_width
      auto elementOffsetInBurst = position;

      // Generate aligned data for each possible offset position
      mlir::Value alignedData;

      if (entryInfo.dataWidth == 64) {
        // Full 64-bit, no padding needed
        alignedData = rawData;
      } else {
        // Generate padded data for each possible bit position and mux
        llvm::SmallVector<mlir::Value, 4> positionedDataValues;

        for (uint32_t elemOffset = 0; elemOffset < elementsPerBurst; ++elemOffset) {
          uint32_t bitShift = elemOffset * entryInfo.dataWidth;
          uint32_t leftPadWidth = 64 - bitShift - entryInfo.dataWidth;
          uint32_t rightPadWidth = bitShift;

          mlir::Value padded = rawData;

          if (leftPadWidth > 0) {
            auto zeroLeftType = UIntType::get(bodyBuilder.getContext(), leftPadWidth);
            auto zeroLeft = bodyBuilder.create<ConstantOp>(loc, zeroLeftType, llvm::APInt(leftPadWidth, 0));
            padded = bodyBuilder.create<CatPrimOp>(
              loc, mlir::ValueRange{zeroLeft.getResult(), padded});
          }

          if (rightPadWidth > 0) {
            auto zeroRightType = UIntType::get(bodyBuilder.getContext(), rightPadWidth);
            auto zeroRight = bodyBuilder.create<ConstantOp>(loc, zeroRightType, llvm::APInt(rightPadWidth, 0));
            padded = bodyBuilder.create<CatPrimOp>(
              loc, mlir::ValueRange{padded, zeroRight.getResult()});
          }

          positionedDataValues.push_back(padded);
        }

        // Mux to select correct positioned data
        alignedData = positionedDataValues[0];
        for (uint32_t i = 1; i < elementsPerBurst; ++i) {
          auto offsetConst = bodyBuilder.create<ConstantOp>(
            loc, u64Type, llvm::APInt(64, i));
          auto isThisOffset = bodyBuilder.create<EQPrimOp>(
            loc, elementOffsetInBurst, offsetConst.getResult());
          alignedData = bodyBuilder.create<MuxPrimOp>(
            loc, isThisOffset, positionedDataValues[i], alignedData);
        }
      }

      // Return aligned data if mine, else 0
      auto zeroData = bodyBuilder.create<ConstantOp>(
        loc, u64Type, llvm::APInt(64, 0));
      auto resultOp = bodyBuilder.create<MuxPrimOp>(
        loc, isMine, alignedData, zeroData.getResult());

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, mlir::ValueRange{resultOp.getResult()});
    });

    burstRead->finalize();

    // burst_write method
    auto *burstWrite = wrapper->addMethod(
      "burst_write",
      {{"addr", u64Type}, {"data", u64Type}},
      {}
    );

    burstWrite->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto u1Type = UIntType::get(guardBuilder.getContext(), 1);
      auto trueConst = guardBuilder.create<ConstantOp>(loc, u1Type, llvm::APInt(1, 1));
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getResult());
    });

    burstWrite->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = args[0];
      auto data = args[1];

      // Helper: Create constants
      auto burstSizeConst = bodyBuilder.create<ConstantOp>(loc, u64Type, llvm::APInt(64, 8));  // 64-bit burst = 8 bytes
      auto numBanksConst = bodyBuilder.create<ConstantOp>(loc, u64Type, llvm::APInt(64, entryInfo.numBanks));
      auto myBankConst = bodyBuilder.create<ConstantOp>(loc, u64Type, llvm::APInt(64, bankIdx));
      auto elementsPerBurstConst = bodyBuilder.create<ConstantOp>(loc, u64Type, llvm::APInt(64, elementsPerBurst));

      // Helper 1: burst_idx = addr / 8
      auto burstIdx = bodyBuilder.create<DivPrimOp>(loc, addr, burstSizeConst.getResult());

      // Helper 2: burst_pattern = burst_idx % num_banks
      auto burstPattern = bodyBuilder.create<RemPrimOp>(loc, burstIdx, numBanksConst.getResult());

      // Helper 3: Calculate position and check participation
      auto posTemp = bodyBuilder.create<SubPrimOp>(loc, myBankConst.getResult(), burstPattern);
      auto positionInSeq = bodyBuilder.create<AddPrimOp>(loc, posTemp, numBanksConst.getResult());
      auto position = bodyBuilder.create<RemPrimOp>(loc, positionInSeq, numBanksConst.getResult());
      auto isMine = bodyBuilder.create<LTPrimOp>(loc, position, elementsPerBurstConst.getResult());

      // Helper 4: Calculate local offset (same as read)
      auto patternOffset = bodyBuilder.create<SubPrimOp>(loc, myBankConst.getResult(), burstPattern);
      auto patternOffsetAdjusted = bodyBuilder.create<AddPrimOp>(loc, patternOffset, numBanksConst.getResult());
      auto patternOffsetMod = bodyBuilder.create<RemPrimOp>(loc, patternOffsetAdjusted, numBanksConst.getResult());
      auto adjustedBurstIdx = bodyBuilder.create<SubPrimOp>(loc, burstIdx, patternOffsetMod);
      auto localOffset = bodyBuilder.create<DivPrimOp>(loc, adjustedBurstIdx, numBanksConst.getResult());

      // Truncate to address width
      auto bankAddrType = UIntType::get(bodyBuilder.getContext(), entryInfo.addrWidth);
      auto localAddrTrunc = bodyBuilder.create<BitsPrimOp>(
        loc, bankAddrType, localOffset, entryInfo.addrWidth - 1, 0);

      // Helper 5: Calculate data slice position
      auto elementOffsetInBurst = position;

      // Extract all possible data slices and mux
      auto bankDataType = UIntType::get(bodyBuilder.getContext(), entryInfo.dataWidth);
      llvm::SmallVector<mlir::Value, 4> dataSlices;

      for (uint32_t elemOffset = 0; elemOffset < elementsPerBurst; ++elemOffset) {
        uint32_t bitStart = elemOffset * entryInfo.dataWidth;
        uint32_t bitEnd = bitStart + entryInfo.dataWidth - 1;
        auto slice = bodyBuilder.create<BitsPrimOp>(
          loc, bankDataType, data, bitEnd, bitStart);
        dataSlices.push_back(slice);
      }

      mlir::Value myData = dataSlices[0];
      for (uint32_t i = 1; i < elementsPerBurst; ++i) {
        auto offsetConst = bodyBuilder.create<ConstantOp>(
          loc, u64Type, llvm::APInt(64, i));
        auto isThisOffset = bodyBuilder.create<EQPrimOp>(
          loc, elementOffsetInBurst, offsetConst.getResult());
        myData = bodyBuilder.create<MuxPrimOp>(
          loc, isThisOffset, dataSlices[i], myData);
      }

      // Write to wire instances using callMethod
      // The wires will be written conditionally based on isMine
      auto u1Type = UIntType::get(bodyBuilder.getContext(), 1);
      auto trueConst = bodyBuilder.create<ConstantOp>(loc, u1Type, llvm::APInt(1, 1));
      auto falseConst = bodyBuilder.create<ConstantOp>(loc, u1Type, llvm::APInt(1, 0));

      // Mux to select enable value based on isMine
      auto enableValue = bodyBuilder.create<MuxPrimOp>(loc, isMine, trueConst.getResult(), falseConst.getResult());

      // Call write methods on wire instances
      writeEnableWire->callMethod("write", {enableValue}, bodyBuilder);
      writeDataWire->callMethod("write", {myData}, bodyBuilder);
      writeAddrWire->callMethod("write", {localAddrTrunc}, bodyBuilder);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    burstWrite->finalize();

    // Create a rule that reads from wires and conditionally writes to bank
    // NOTE: This currently crashes because callValue on regular CMT2 Module instances
    // returns empty results. This is a bug in ECMT2 Instance.cpp that needs to be fixed.
    auto *writeRule = wrapper->addRule("do_bank_write");

    writeRule->guard([&](mlir::OpBuilder &guardBuilder) {
      // Guard: check if enable wire is 1 (compare wire value with 1)
      auto u1Type = UIntType::get(guardBuilder.getContext(), 1);
      auto oneConst = guardBuilder.create<ConstantOp>(loc, u1Type, llvm::APInt(1, 1));

      // BUG: writeEnableWire->callValue returns empty vector for regular Module instances
      auto enableValues = writeEnableWire->callValue("read", guardBuilder);
      auto isEnabled = guardBuilder.create<EQPrimOp>(loc, enableValues[0], oneConst.getResult());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, isEnabled.getResult());
    });

    writeRule->body([&, bankInst](mlir::OpBuilder &bodyBuilder) {
      // Read data and address from wires and write to bank
      auto dataValues = writeDataWire->callValue("read", bodyBuilder);
      auto addrValues = writeAddrWire->callValue("read", bodyBuilder);

      bankInst->callMethod("write", {dataValues[0], addrValues[0]}, bodyBuilder);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    writeRule->finalize();

    return wrapper;
  }

  /// Generate memory entry submodule for a single memory entry
  Module *generateMemoryEntryModule(const MemoryEntryInfo &entryInfo,
                                    Circuit &circuit,
                                    Clock clk, Reset rst,
                                    const llvm::SmallVector<std::string, 4> &bankNames) {
    llvm::outs() << "  Creating submodule for entry: " << entryInfo.name << "\n";

    // Validate configuration to avoid bank conflicts in burst access
    // Constraint: num_banks * data_width >= 64
    // If < 64, multiple elements in a 64-bit burst map to the same bank (conflict!)
    if (entryInfo.isCyclic) {
      uint32_t totalBankWidth = entryInfo.numBanks * entryInfo.dataWidth;
      if (totalBankWidth < 64) {
        llvm::errs() << "  ERROR: Cyclic partition configuration causes bank conflicts!\n";
        llvm::errs() << "    Entry: " << entryInfo.name << "\n";
        llvm::errs() << "    Config: " << entryInfo.numBanks << " banks × "
                     << entryInfo.dataWidth << " bits = " << totalBankWidth << " bits\n";
        llvm::errs() << "    A 64-bit burst contains " << (64 / entryInfo.dataWidth)
                     << " elements\n";
        llvm::errs() << "    Elements per bank: "
                     << ((64 / entryInfo.dataWidth + entryInfo.numBanks - 1) / entryInfo.numBanks)
                     << " (CONFLICT!)\n";
        llvm::errs() << "    Requirement: num_banks × data_width >= 64\n";
        llvm::errs() << "    Valid examples: 8×8, 4×16, 2×32, 1×64, 4×32, 8×16, etc.\n";
      } else if (totalBankWidth > 64) {
        llvm::outs() << "    Note: Each 64-bit burst writes to " << (64 / entryInfo.dataWidth)
                     << " of " << entryInfo.numBanks << " banks (no conflict)\n";
      }
    }

    // Create submodule for this memory entry
    std::string moduleName = "memory_" + entryInfo.name;
    auto *entryModule = circuit.addModule(moduleName);

    // Add clock and reset arguments
    (void)clk;  // Will be used when creating instances
    (void)rst;
    Clock subClk = entryModule->addClockArgument("clk");
    Reset subRst = entryModule->addResetArgument("rst");

    auto &builder = entryModule->getBuilder();
    auto loc = entryModule->getLoc();

    auto u64Type = UIntType::get(builder.getContext(), 64);

    // Create external module declaration at circuit level (BEFORE creating instances)
    // Save insertion point first
    auto savedIPForExtMod = builder.saveInsertionPoint();

    llvm::StringMap<int64_t> memParams;
    memParams["data_width"] = entryInfo.dataWidth;
    memParams["addr_width"] = entryInfo.addrWidth;
    memParams["depth"] = entryInfo.depth;

    auto *memMod = circuit.addExternalModule("Mem1r1w", memParams);

    // Restore insertion point back to entry module
    builder.restoreInsertionPoint(savedIPForExtMod);

    // Bind clock and reset
    memMod->bindClock("clk", "clock")
          .bindReset("rst", "reset");

    // Bind memory methods
    memMod->bindMethod("rd0", "en", "", {"raddr"}, {});
    memMod->bindValue("rd1", "rd1_valid", {"rdata"});
    memMod->bindMethod("write", "wen", "", {"wdata", "waddr"}, {});

    // Add conflict relationships
    memMod->addConflict("write", "write");
    memMod->addConflict("rd0", "rd0");

    // Save insertion point BEFORE creating wrapper modules
    // (generateBankWrapperModule will change the insertion point)
    auto savedIP = builder.saveInsertionPoint();

    // Create bank wrapper modules (this changes insertion point to each wrapper)
    llvm::SmallVector<Module*, 4> wrapperModules;
    for (size_t i = 0; i < entryInfo.numBanks; ++i) {
      auto *wrapperMod = generateBankWrapperModule(entryInfo, circuit, i, memMod, subClk, subRst);
      wrapperModules.push_back(wrapperMod);
    }

    // Restore insertion point back to entryModule before adding instances/methods
    builder.restoreInsertionPoint(savedIP);

    // Create wrapper instances IN THIS SUBMODULE
    llvm::SmallVector<Instance*, 4> wrapperInstances;
    for (size_t i = 0; i < entryInfo.numBanks; ++i) {
      std::string wrapperName = "bank_wrap_" + std::to_string(i);
      auto *wrapperInst = entryModule->addInstance(wrapperName, wrapperModules[i],
                                                   {subClk.getValue(), subRst.getValue()});
      wrapperInstances.push_back(wrapperInst);
    }

    llvm::outs() << "    Creating burst_read method for " << entryInfo.name << "\n";
    llvm::outs() << "    Number of banks: " << entryInfo.numBanks << "\n";

    // Create burst_read method: forwards addr to all banks, ORs results
    auto *burstRead = entryModule->addMethod(
      "burst_read",
      {{"addr", u64Type}},
      {u64Type}
    );

    // Guard: always ready (address range checked at parent level)
    burstRead->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto u1Type = UIntType::get(guardBuilder.getContext(), 1);
      auto trueConst = guardBuilder.create<ConstantOp>(loc, u1Type, llvm::APInt(1, 1));
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getResult());
    });

    // Body: Simple OR aggregation of all bank wrapper outputs
    // Each wrapper returns aligned 64-bit data if address is for that bank, else 0
    burstRead->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = args[0];

      llvm::outs() << "    DEBUG: burst_read body for " << entryInfo.name << "\n";
      llvm::outs() << "    DEBUG: Number of wrapper instances: " << wrapperInstances.size() << "\n";

      // Use CallOp to call wrapper methods (Instance::callMethod doesn't work for methods with return values)
      auto calleeSymbol0 = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), wrapperInstances[0]->getName());
      auto methodSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_read");
      auto callOp0 = bodyBuilder.create<circt::cmt2::CallOp>(
        loc, mlir::TypeRange{u64Type}, mlir::ValueRange{addr},
        calleeSymbol0, methodSymbol, nullptr, nullptr);
      mlir::Value result = callOp0.getResult(0);

      // OR together all other wrapper outputs
      for (size_t i = 1; i < entryInfo.numBanks; ++i) {
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), wrapperInstances[i]->getName());
        auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{u64Type}, mlir::ValueRange{addr},
          calleeSymbol, methodSymbol, nullptr, nullptr);
        auto data = callOp.getResult(0);
        result = bodyBuilder.create<OrPrimOp>(loc, result, data);
      }

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, result);
    });

    burstRead->finalize();
    llvm::outs() << "    Finalized burst_read method for " << entryInfo.name << "\n";

    llvm::outs() << "    Creating burst_write method for " << entryInfo.name << "\n";

    // Create burst_write method: forwards data to all banks
    auto *burstWrite = entryModule->addMethod(
      "burst_write",
      {{"addr", u64Type}, {"data", u64Type}},
      {}
    );

    burstWrite->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto u1Type = UIntType::get(guardBuilder.getContext(), 1);
      auto trueConst = guardBuilder.create<ConstantOp>(loc, u1Type, llvm::APInt(1, 1));
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getResult());
    });

    burstWrite->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = args[0];
      auto data = args[1];

      // Simple broadcast to all bank wrappers using CallOp
      // Each wrapper decides if it should write based on address
      auto methodSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_write");
      for (size_t i = 0; i < entryInfo.numBanks; ++i) {
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), wrapperInstances[i]->getName());
        bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{}, mlir::ValueRange{addr, data},
          calleeSymbol, methodSymbol, nullptr, nullptr);
      }

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    burstWrite->finalize();
    llvm::outs() << "    Finalized burst_write method for " << entryInfo.name << "\n";

    return entryModule;
  }

  /// Generate burst access logic (address decoding and bank selection)
  void generateBurstAccessLogic(Module *poolModule,
                                const llvm::SmallVector<MemoryEntryInfo, 4> &memEntryInfos,
                                Circuit &circuit,
                                Clock clk, Reset rst) {

    llvm::outs() << "\n=== Generating Burst Access Logic ===\n";

    auto &builder = poolModule->getBuilder();
    auto loc = poolModule->getLoc();
    auto u64Type = UIntType::get(builder.getContext(), 64);

    // Step 1: Create all memory entry submodules
    // Save the insertion point first, since circuit.addModule() will change it
    auto savedIP = builder.saveInsertionPoint();

    llvm::SmallVector<Module*, 4> entryModules;
    for (const auto &entryInfo : memEntryInfos) {
      // Collect bank names
      llvm::SmallVector<std::string, 4> bankNames;
      for (size_t i = 0; i < entryInfo.numBanks; ++i) {
        bankNames.push_back(entryInfo.name + "_bank" + std::to_string(i));
      }

      // Create submodule for this memory entry (this will change insertion point)
      auto *entryMod = generateMemoryEntryModule(entryInfo, circuit, clk, rst, bankNames);
      entryModules.push_back(entryMod);
    }

    // Step 2: Restore insertion point back to poolModule's body before creating instances
    builder.restoreInsertionPoint(savedIP);

    llvm::SmallVector<Instance*, 4> entryInstances;
    for (size_t i = 0; i < memEntryInfos.size(); ++i) {
      const auto &entryInfo = memEntryInfos[i];
      auto *entryMod = entryModules[i];

      std::string instanceName = "inst_" + entryInfo.name;
      llvm::outs() << "  Creating instance '" << instanceName << "' in ScratchpadMemoryPool\n";

      auto *entryInst = poolModule->addInstance(instanceName, entryMod,
                                                {clk.getValue(), rst.getValue()});
      entryInstances.push_back(entryInst);
    }

    llvm::outs() << "\n  Creating top-level burst methods in pool module\n";

    // Create top-level burst_read method that dispatches to correct memory entry
    auto *topBurstRead = poolModule->addMethod(
      "burst_read",
      {{"addr", u64Type}},
      {u64Type}
    );

    topBurstRead->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto u1Type = UIntType::get(guardBuilder.getContext(), 1);
      auto trueConst = guardBuilder.create<ConstantOp>(loc, u1Type, llvm::APInt(1, 1));
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getResult());
    });

    topBurstRead->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = args[0];

      // Accumulate results from all memory entries (OR them together)
      auto resultInit = bodyBuilder.create<ConstantOp>(
        loc, u64Type, llvm::APInt(64, 0));
      mlir::Value result = resultInit.getResult();

      for (size_t i = 0; i < memEntryInfos.size(); ++i) {
        const auto &entryInfo = memEntryInfos[i];
        auto *entryInst = entryInstances[i];

        uint32_t entryStart = entryInfo.baseAddress;
        uint32_t entryEnd = entryStart + entryInfo.bankSize;

        // Calculate relative address for this entry
        auto startConst = bodyBuilder.create<ConstantOp>(
          loc, u64Type, llvm::APInt(64, entryStart));
        auto relAddrWide = bodyBuilder.create<SubPrimOp>(loc, addr, startConst.getResult());

        // SubPrimOp produces uint<65>, truncate back to uint<64>
        auto relAddr = bodyBuilder.create<BitsPrimOp>(loc, u64Type, relAddrWide, 63, 0);

        // Check if address is in this entry's range
        auto endConst = bodyBuilder.create<ConstantOp>(
          loc, u64Type, llvm::APInt(64, entryEnd));

        auto geStart = bodyBuilder.create<GEQPrimOp>(loc, addr, startConst.getResult());
        auto ltEnd = bodyBuilder.create<LTPrimOp>(loc, addr, endConst.getResult());
        auto inRange = bodyBuilder.create<AndPrimOp>(loc, geStart, ltEnd);

        // Call submodule's burst_read (manual CallOp needed for methods with return values)
        llvm::outs() << "DEBUG: Calling burst_read on entry " << entryInfo.name << "\n";
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), entryInst->getName());
        auto methodSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_read");
        auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{u64Type}, mlir::ValueRange{relAddr},
          calleeSymbol, methodSymbol, nullptr, nullptr);
        llvm::outs() << "DEBUG: burst_read returned " << callOp.getNumResults() << " results\n";
        if (callOp.getNumResults() == 0) {
          llvm::errs() << "ERROR: CallOp for burst_read returned no results!\n";
          llvm::errs() << "  Instance: " << entryInst->getName() << "\n";
        }
        auto entryResult = callOp.getResult(0);

        // Use mux to conditionally include this result
        auto zeroData = bodyBuilder.create<ConstantOp>(
          loc, u64Type, llvm::APInt(64, 0));
        auto selectedData = bodyBuilder.create<MuxPrimOp>(
          loc, inRange, entryResult, zeroData.getResult());

        // OR into final result
        result = bodyBuilder.create<OrPrimOp>(loc, result, selectedData);
      }

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, result);
    });

    topBurstRead->finalize();
    llvm::outs() << "  Finalized top-level burst_read method\n";

    // Create top-level burst_write method that dispatches to correct memory entry
    auto *topBurstWrite = poolModule->addMethod(
      "burst_write",
      {{"addr", u64Type}, {"data", u64Type}},
      {}  // No return value
    );

    topBurstWrite->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto u1Type = UIntType::get(guardBuilder.getContext(), 1);
      auto trueConst = guardBuilder.create<ConstantOp>(loc, u1Type, llvm::APInt(1, 1));
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getResult());
    });

    topBurstWrite->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = args[0];
      auto data = args[1];

      // Dispatch to all memory entries based on address range
      for (size_t i = 0; i < memEntryInfos.size(); ++i) {
        const auto &entryInfo = memEntryInfos[i];
        auto *entryInst = entryInstances[i];

        uint32_t entryStart = entryInfo.baseAddress;

        // Calculate relative address for this entry
        auto startConst = bodyBuilder.create<ConstantOp>(
          loc, u64Type, llvm::APInt(64, entryStart));
        auto relAddrWide = bodyBuilder.create<SubPrimOp>(loc, addr, startConst.getResult());

        // SubPrimOp produces uint<65>, truncate back to uint<64>
        auto relAddr = bodyBuilder.create<BitsPrimOp>(loc, u64Type, relAddrWide, 63, 0);

        // Call submodule's burst_write
        // Note: We call all submodules, but only the one with matching address range will write
        // (the submodule's internal logic handles address decoding)
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), entryInst->getName());
        auto methodSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_write");
        bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{}, mlir::ValueRange{relAddr, data},
          calleeSymbol, methodSymbol, nullptr, nullptr);
      }

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    topBurstWrite->finalize();
    llvm::outs() << "  Finalized top-level burst_write method\n";

    llvm::outs() << "=== Burst Access Logic Generation Complete ===\n";
  }

  /// Generate the CMT2 memory pool module
  void generateMemoryPool(ModuleOp moduleOp, aps::MemoryMapOp memoryMapOp) {
    MLIRContext *context = moduleOp.getContext();
    OpBuilder builder(context);

    // Collect all memory entries
    llvm::SmallVector<aps::MemEntryOp, 4> memEntries;
    memoryMapOp.getRegion().walk([&](aps::MemEntryOp entry) {
      memEntries.push_back(entry);
    });

    if (memEntries.empty()) {
      return;
    }

    // Create CMT2 circuit
    Circuit circuit("MemoryPool", *context);

    // Generate memory pool module
    auto *poolModule = circuit.addModule("ScratchpadMemoryPool");

    // Add clock and reset
    Clock clk = poolModule->addClockArgument("clk");
    Reset rst = poolModule->addResetArgument("rst");

    // Store instances for each memory entry (for address decoding)
    llvm::SmallVector<MemoryEntryInfo, 4> memEntryInfos;

    // For each memory entry, create SRAM banks
    for (auto memEntry : memEntries) {
      auto bankSymbols = memEntry.getBankSymbols();
      uint32_t numBanks = memEntry.getNumBanks();
      uint32_t baseAddress = memEntry.getBaseAddress();
      uint32_t bankSize = memEntry.getBankSize();
      uint32_t cyclic = memEntry.getCyclic();

      llvm::errs() << "Generating memory entry: " << memEntry.getName()
                   << " with " << numBanks << " banks\n";
      llvm::errs() << "  Base address: 0x" << llvm::format("%x", baseAddress) << "\n";
      llvm::errs() << "  Bank size: " << bankSize << " bytes\n";
      llvm::errs() << "  Partition mode: " << (cyclic ? "cyclic" : "block") << "\n";

      // Get parameters from the first bank (all banks should have same type)
      if (bankSymbols.empty()) {
        llvm::errs() << "Error: Memory entry " << memEntry.getName()
                     << " has no banks\n";
        continue;
      }

      auto firstBankSymAttr = llvm::dyn_cast<FlatSymbolRefAttr>(bankSymbols[0]);
      if (!firstBankSymAttr) {
        llvm::errs() << "Error: Invalid bank symbol attribute\n";
        continue;
      }

      // Look up the memref.global for this bank
      auto globalOp = getGlobalMemRef(moduleOp, firstBankSymAttr.getValue());
      if (!globalOp) {
        llvm::errs() << "Error: Could not find memref.global for bank "
                     << firstBankSymAttr.getValue() << "\n";
        continue;
      }

      // Extract memory parameters from the memref type
      int dataWidth = 32;  // defaults
      int addrWidth = 10;
      int depth = 1024;

      if (!extractMemoryParameters(globalOp, dataWidth, addrWidth, depth)) {
        llvm::errs() << "Error: Could not extract memory parameters from "
                     << firstBankSymAttr.getValue() << ", using defaults\n";
      } else {
        llvm::errs() << "  Data width: " << dataWidth << " bits\n";
        llvm::errs() << "  Address width: " << addrWidth << " bits\n";
        llvm::errs() << "  Depth: " << depth << " entries\n";
      }

      // Create MemoryEntryInfo for this entry
      // Note: Bank instances will be created inside submodules, not here
      MemoryEntryInfo entryInfo;
      entryInfo.name = memEntry.getName().str();
      entryInfo.baseAddress = baseAddress;
      entryInfo.bankSize = bankSize;
      entryInfo.numBanks = numBanks;
      entryInfo.isCyclic = (cyclic != 0);
      entryInfo.dataWidth = dataWidth;
      entryInfo.addrWidth = addrWidth;
      entryInfo.depth = depth;

      // Add this memory entry info to the list
      memEntryInfos.push_back(std::move(entryInfo));
    }

    // Now generate address decoding logic for burst access
    // This will create memory entry submodules with bank instances
    generateBurstAccessLogic(poolModule, memEntryInfos, circuit, clk, rst);

    // Emit the generated CMT2 MLIR
    builder.setInsertionPointToEnd(moduleOp.getBody());

    std::string mlirStr = circuit.emitMLIRString();
    llvm::errs() << "\n=== Generated Memory Pool (CMT2) ===\n";
    llvm::outs() << mlirStr << "\n";

    // TODO: Parse and insert the generated MLIR into the module
    // For now, just print it for inspection
  }
};

} // namespace mlir

std::unique_ptr<mlir::Pass> mlir::createAPSMemoryPoolGenPass() {
  return std::make_unique<mlir::APSMemoryPoolGenPass>();
}
