//===- APSToCMT2GenPass.cpp - Generate CMT2 hardware from APS TOR --------===//
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
#include "TOR/TOR.h"
#include "circt/Dialect/Cmt2/ECMT2/Circuit.h"
#include "circt/Dialect/Cmt2/ECMT2/FunctionLike.h"
#include "circt/Dialect/Cmt2/ECMT2/Instance.h"
#include "circt/Dialect/Cmt2/ECMT2/Module.h"
#include "circt/Dialect/Cmt2/ECMT2/ModuleLibrary.h"
#include "circt/Dialect/Cmt2/ECMT2/STLLibrary.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "circt/Dialect/Cmt2/Cmt2Dialect.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir-c/Support.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "aps-memory-pool-gen"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

// Data structures for TOR to CMT2 conversion (from rulegenpass.cpp)
struct SlotInfo {
  SmallVector<Operation *, 4> ops;
};

struct CrossUseInfo {
  mlir::Value producerValue;
  int64_t producerSlot = 0;
  Operation *consumerOp = nullptr;
  unsigned operandIndex = 0;
  int64_t consumerSlot = 0;
  std::string instanceName;
  mlir::Type firType;
  Instance *wireInstance = nullptr;
};

struct APSToCMT2GenPass : public PassWrapper<APSToCMT2GenPass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(APSToCMT2GenPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<aps::APSDialect>();
    registry.insert<circt::cmt2::Cmt2Dialect>();
    registry.insert<FIRRTLDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    llvm::outs() << "DEBUG: APSToCMT2GenPass::runOnOperation() started\n";

    // Find the aps.memorymap operation
    aps::MemoryMapOp memoryMapOp;
    moduleOp.walk([&](aps::MemoryMapOp op) {
      memoryMapOp = op;
      return WalkResult::interrupt();
    });

    // Initialize CMT2 module library
    auto &library = ModuleLibrary::getInstance();
    llvm::outs() << "DEBUG: Initializing CMT2 module library\n";

    // Try to find the module library manifest
    // First try: relative to build directory
    llvm::SmallString<256> manifestPath;
    llvm::sys::path::append(manifestPath, "circt/lib/Dialect/Cmt2/ModuleLibrary/manifest.yaml");

    // Set library base path (parent of manifest)
    if (llvm::sys::fs::exists(manifestPath)) {
      llvm::outs() << "DEBUG: Found manifest at " << manifestPath << "\n";
      llvm::SmallString<256> libraryPath = llvm::sys::path::parent_path(manifestPath);
      library.setLibraryPath(libraryPath);

      // Load the manifest
      if (mlir::failed(library.loadManifest(manifestPath))) {
        llvm::outs() << "DEBUG: Failed to load module library manifest\n";
        moduleOp.emitWarning() << "Failed to load module library manifest from "
                               << manifestPath;
        return;
      }
      llvm::outs() << "DEBUG: Successfully loaded module library manifest\n";
    } else {
      llvm::outs() << "DEBUG: Module library manifest not found at " << manifestPath << "\n";
      moduleOp.emitWarning() << "Module library manifest not found. "
                             << "External FIRRTL modules may not work correctly.";
      return;
    }

    // Create CMT2 circuit (always needed for both memory and rules)
    llvm::outs() << "DEBUG: Creating CMT2 circuit\n";
    MLIRContext *context = moduleOp.getContext();
    Circuit circuit("MemoryPool", *context);

    // Generate the memory pool
    llvm::outs() << "DEBUG: About to call generateMemoryPool\n";

    Module *poolModule = generateMemoryPool(circuit, moduleOp, memoryMapOp);

    // Generate rule-based main module for TOR functions
    auto [mainModule, poolInstance] = generateRuleBasedMainModule(moduleOp, circuit, poolModule);

    // Add burst read/write methods to expose memory pool functionality
    addBurstMethodsToMainModule(mainModule, poolInstance);
    
    auto generatedModule = circuit.generateMLIR();
    if (!generatedModule) {
      moduleOp.emitError() << "failed to materialize MLIR";
      return;
    }

    // Preserve all original Ops
    auto &targetBlock = moduleOp.getBodyRegion().front();
    auto &generatedOps = generatedModule->getBodyRegion().front().getOperations();

    for (auto &op : llvm::make_early_inc_range(generatedOps)) {
      if (op.hasTrait<mlir::OpTrait::IsTerminator>())
        continue;
      targetBlock.push_back(op.clone());
    }
  }

  StringRef getArgument() const final { return "aps-to-cmt2-gen"; }
  StringRef getDescription() const final {
    return "Generate CMT2 hardware from APS TOR operations";
  }

private:
  /// Get memref.global by symbol name
  memref::GlobalOp getGlobalMemRef(mlir::Operation *scope, StringRef symbolName) {
    if (!scope)
      return nullptr;
    auto symRef = mlir::FlatSymbolRefAttr::get(scope->getContext(), symbolName);
    if (auto *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(scope, symRef))
      return mlir::dyn_cast<memref::GlobalOp>(symbol);
    return nullptr;
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

  // ============================================================================
  //
  //                                BANK WRAPPER
  //
  // ============================================================================

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
    auto savedIPForWires = builder.saveInsertionPoint();

    auto wireEnableMod = STLLibrary::createWireDefaultModule(1, 0, circuit);
    auto wireDataMod = STLLibrary::createWireModule(entryInfo.dataWidth, circuit);
    auto wireAddrMod = STLLibrary::createWireModule(entryInfo.addrWidth, circuit);

    // Restore insertion point back to wrapper
    builder.restoreInsertionPoint(savedIPForWires);

    // Create wire_default instances in the wrapper
    auto *writeEnableWire = wrapper->addInstance("write_enable_wire", wireEnableMod, {});
    auto *writeDataWire = wrapper->addInstance("write_data_wire", wireDataMod, {});
    auto *writeAddrWire = wrapper->addInstance("write_addr_wire", wireAddrMod, {});

    // burst_read method: returns 64-bit aligned data if address is for this bank, else 0
    auto *burstRead = wrapper->addMethod("burst_read", {{"addr", u64Type}}, {u64Type});

    burstRead->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueVal = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
    });

    burstRead->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());

      auto burstSizeConst = UInt::constant(8, 64, bodyBuilder, wrapper->getLoc()); // 64-bit burst = 8 bytes
      auto numBanksConst = UInt::constant(entryInfo.numBanks, 64, bodyBuilder, wrapper->getLoc());
      auto myBankConst = UInt::constant(bankIdx, 64, bodyBuilder, wrapper->getLoc());
      auto elementsPerBurstConst = UInt::constant(64, 64, bodyBuilder, wrapper->getLoc());
      
      // burst_idx = addr / 8
      auto burstIdx = addr / burstSizeConst;
      // burst_pattern = burst_idx % num_banks
      auto burstPattern = burstIdx / numBanksConst;

      // Check if this bank participates: isMine = (burst_pattern <= my_bank) && (my_bank < burst_pattern + elements_per_burst)
      // Simplified for cyclic: position = (my_bank - burst_pattern + num_banks) % num_banks
      auto position = (myBankConst - burstPattern + numBanksConst) % numBanksConst;
      auto isMine = position < elementsPerBurstConst;

      // Calculate local offset
      // For banks in pattern: offset = (burst_idx - pattern_offset) / num_banks
      // pattern_offset = my_bank - burst_pattern (when my_bank >= burst_pattern) or my_bank + num_banks - burst_pattern (wrapped)
      // Simplified: offset = burst_idx / num_banks (for pattern 0), adjust for other patterns
      auto patternOffsetMod = (myBankConst - burstPattern + numBanksConst) % numBanksConst;

      // adjusted_burst_idx = burst_idx - pattern_offset_mod
      auto localOffset = (burstIdx - patternOffsetMod) / numBanksConst;
      auto localAddrTrunc = localOffset.bits(entryInfo.addrWidth - 1, 0);

      auto bankDataValues = bankInst->callMethod("read", {localAddrTrunc.getValue()}, bodyBuilder);
      auto rawData = Signal(bankDataValues[0], &bodyBuilder, loc);

      // Helper 5: Calculate bit position: position * data_width
      auto elementOffsetInBurst = position;

      // Generate aligned data for each possible offset position
      mlir::Value alignedData;

      if (entryInfo.dataWidth == 64) {
        // Full 64-bit, no padding needed
        alignedData = rawData.getValue();
      } else {
        // Generate padded data for each possible bit position and mux
        llvm::SmallVector<mlir::Value, 4> positionedDataValues;

        for (uint32_t elemOffset = 0; elemOffset < elementsPerBurst; ++elemOffset) {
          uint32_t bitShift = elemOffset * entryInfo.dataWidth;
          uint32_t leftPadWidth = 64 - bitShift - entryInfo.dataWidth;
          uint32_t rightPadWidth = bitShift;
          auto padded = rawData;
          if (leftPadWidth > 0) {
            auto zeroLeft = UInt::constant(0, leftPadWidth, bodyBuilder, loc);
            padded = zeroLeft.cat(padded);
          }
          if (rightPadWidth > 0) {
            auto zeroRight = UInt::constant(0, rightPadWidth, bodyBuilder, loc);
            padded = padded.cat(zeroRight);
          }
          positionedDataValues.push_back(padded.getValue());
        }

        // Mux to select correct positioned data
        alignedData = positionedDataValues[0];
        for (uint32_t i = 1; i < elementsPerBurst; ++i) {
          auto offsetConst = UInt::constant(i,  64, bodyBuilder, loc);
          auto isThisOffset = elementOffsetInBurst == offsetConst;
          alignedData = isThisOffset.mux(Signal(positionedDataValues[i], &bodyBuilder, loc),
            Signal(alignedData, &bodyBuilder, loc)).getValue();
        }
      }

      // Return aligned data if mine, else 0
      auto zeroData = UInt::constant(0, 64, bodyBuilder, wrapper->getLoc());
      auto resultOp = isMine.mux(Signal(alignedData, &bodyBuilder, wrapper->getLoc()), zeroData);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, mlir::ValueRange{resultOp.getValue()});
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
      auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());
      auto data = Signal(args[1], &bodyBuilder, wrapper->getLoc());

      // Helper: Create constants using Signal
      auto burstSizeConst = UInt::constant(8, 64, bodyBuilder, wrapper->getLoc());  // 64-bit burst = 8 bytes
      auto numBanksConst = UInt::constant(entryInfo.numBanks, 64, bodyBuilder, wrapper->getLoc());
      auto myBankConst = UInt::constant(bankIdx, 64, bodyBuilder, wrapper->getLoc());
      auto elementsPerBurstConst = UInt::constant(elementsPerBurst, 64, bodyBuilder, wrapper->getLoc());

      // Helper 1: burst_idx = addr / 8
      auto burstIdx = addr / burstSizeConst;

      // Helper 2: burst_pattern = burst_idx % num_banks
      auto burstPattern = burstIdx % numBanksConst;

      // Helper 3: Calculate position and check participation
      auto position = ((myBankConst - burstPattern + numBanksConst) % numBanksConst);
      auto isMine = position < elementsPerBurstConst;

      // Helper 4: Calculate local offset (same as read)
      auto patternOffsetMod = (myBankConst - burstPattern + numBanksConst) % numBanksConst;
      auto localOffset = (burstIdx - patternOffsetMod) / numBanksConst;
      auto localAddrTrunc = localOffset.bits(entryInfo.addrWidth - 1, 0);

      // Helper 5: Calculate data slice position
      auto elementOffsetInBurst = position;

      // Extract all possible data slices and mux using Signal
      llvm::SmallVector<Signal, 4> dataSlices;

      for (uint32_t elemOffset = 0; elemOffset < elementsPerBurst; ++elemOffset) {
        uint32_t bitStart = elemOffset * entryInfo.dataWidth;
        uint32_t bitEnd = bitStart + entryInfo.dataWidth - 1;
        auto slice = data.bits(bitEnd, bitStart);
        dataSlices.push_back(slice);
      }

      auto myData = dataSlices[0];
      for (uint32_t i = 1; i < elementsPerBurst; ++i) {
        auto offsetConst = UInt::constant(i, 64, bodyBuilder, wrapper->getLoc());
        auto isThisOffset = (elementOffsetInBurst == offsetConst);
        myData = isThisOffset.mux(dataSlices[i], myData);
      }

      // Write to wire instances using callMethod
      // The wires will be written conditionally based on isMine
      auto trueConst = UInt::constant(1, 1, bodyBuilder, wrapper->getLoc());
      auto falseConst = UInt::constant(0, 1, bodyBuilder, wrapper->getLoc());

      // Mux to select enable value based on isMine
      auto enableValue = isMine.mux(trueConst, falseConst);

      // Call write methods on wire instances
      writeEnableWire->callMethod("write", {enableValue.getValue()}, bodyBuilder);
      writeDataWire->callMethod("write", {myData.getValue()}, bodyBuilder);
      writeAddrWire->callMethod("write", {localAddrTrunc.getValue()}, bodyBuilder);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    burstWrite->finalize();

    // Direct bank-level read method (no burst translation), exposes native port
    auto bankAddrType = UIntType::get(builder.getContext(), entryInfo.addrWidth);
    auto bankDataType = UIntType::get(builder.getContext(), entryInfo.dataWidth);

    auto *directReadMethod = wrapper->addMethod(
      "bank_read",
      {{"addr", bankAddrType}},
      {bankDataType}
    );

    directReadMethod->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    directReadMethod->body([&, bankInst](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());

      // Directly drive the memory bank read ports.
      auto readValues = bankInst->callMethod("read", {addr.getValue()}, bodyBuilder);
      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, readValues[0]);
    });

    directReadMethod->finalize();

    auto *directWriteMethod = wrapper->addMethod(
      "bank_write",
      {{"addr", bankAddrType}, {"data", bankDataType}},
      {}
    );

    directWriteMethod->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    directWriteMethod->body([&, bankInst](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());
      auto data = Signal(args[1], &bodyBuilder, wrapper->getLoc());

      // Directly drive the memory bank write ports.
      bankInst->callMethod("write", {data.getValue(), addr.getValue()}, bodyBuilder);
      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    directWriteMethod->finalize();

    // Create a rule that reads from wires and conditionally writes to bank
    // NOTE: This currently crashes because callValue on regular CMT2 Module instances
    // returns empty results. This is a bug in ECMT2 Instance.cpp that needs to be fixed.
    auto *writeRule = wrapper->addRule("do_bank_write");

    writeRule->guard([&](mlir::OpBuilder &guardBuilder) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
      auto enableValues = Signal(writeEnableWire->callValue("read", guardBuilder)[0], &guardBuilder, loc);
      auto isEnabled = enableValues == trueConst;
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, isEnabled.getValue());
    });

    writeRule->body([&, bankInst](mlir::OpBuilder &bodyBuilder) {
      // Read data and address from wires and write to bank
      auto dataValues = writeDataWire->callValue("read", bodyBuilder);
      auto addrValues = writeAddrWire->callValue("read", bodyBuilder);

      bankInst->callMethod("write", {dataValues[0], addrValues[0]}, bodyBuilder);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    writeRule->finalize();

    // Ensure burst accesses get scheduled ahead of direct bank accesses.
    wrapper->setPrecedence({
      {"burst_read", "bank_read"},
      {"burst_write", "bank_write"}
    });

    return wrapper;
  }

  // ============================================================================
  //
  //                   MEMORY ENTRY (CONTAIN BANKWRAPPER)
  //
  // ============================================================================

  /// Generate memory entry submodule for a single memory entry
  Module *generateMemoryEntryModule(const MemoryEntryInfo &entryInfo,
                                    Circuit &circuit,
                                    Clock clk, Reset rst,
                                    const llvm::SmallVector<std::string, 4> &bankNames) {
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
      }
    }

    // Create submodule for this memory entry
    std::string moduleName = "memory_" + entryInfo.name;
    auto *entryModule = circuit.addModule(moduleName);

    // Add clock and reset arguments
    Clock subClk = entryModule->addClockArgument("clk");
    Reset subRst = entryModule->addResetArgument("rst");

    auto &builder = entryModule->getBuilder();
    auto loc = entryModule->getLoc();

    auto u64Type = UIntType::get(builder.getContext(), 64);

    // Create external module declaration at circuit level (BEFORE creating instances)
    // Save insertion point first
    auto savedIP = builder.saveInsertionPoint();

    llvm::StringMap<int64_t> memParams;
    memParams["data_width"] = entryInfo.dataWidth;
    memParams["addr_width"] = entryInfo.addrWidth;
    memParams["depth"] = entryInfo.depth;

    auto memMod = STLLibrary::createMem1r1w0cModule(entryInfo.dataWidth, entryInfo.addrWidth, entryInfo.depth, circuit);

    // Restore insertion point back to entry module
    builder.restoreInsertionPoint(savedIP);

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

    // Create burst_read method: forwards addr to all banks, ORs results
    auto *burstRead = entryModule->addMethod(
      "burst_read",
      {{"addr", u64Type}},
      {u64Type}
    );

    // Guard: always ready (address range checked at parent level)
    burstRead->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    // Body: Simple OR aggregation of all bank wrapper outputs
    // Each wrapper returns aligned 64-bit data if address is for that bank, else 0
    burstRead->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());

      // Use CallOp to call wrapper methods (Instance::callMethod doesn't work for methods with return values)
      auto calleeSymbol0 = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), wrapperInstances[0]->getName());
      auto methodSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_read");
      auto callOp0 = bodyBuilder.create<circt::cmt2::CallOp>(
        loc, mlir::TypeRange{u64Type}, mlir::ValueRange{addr.getValue()},
        calleeSymbol0, methodSymbol, nullptr, nullptr);
      auto result = Signal(callOp0.getResult(0), &bodyBuilder, entryModule->getLoc());

      // OR together all other wrapper outputs
      for (size_t i = 1; i < entryInfo.numBanks; ++i) {
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), wrapperInstances[i]->getName());
        auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{u64Type}, mlir::ValueRange{addr.getValue()},
          calleeSymbol, methodSymbol, nullptr, nullptr);
        auto data = Signal(callOp.getResult(0), &bodyBuilder, entryModule->getLoc());
        result = result | data;
      }

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, result.getValue());
    });

    burstRead->finalize();
    // Create burst_write method: forwards data to all banks
    auto *burstWrite = entryModule->addMethod(
      "burst_write",
      {{"addr", u64Type}, {"data", u64Type}},
      {}
    );

    burstWrite->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    burstWrite->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());
      auto data = Signal(args[1], &bodyBuilder, entryModule->getLoc());

      // Simple broadcast to all bank wrappers using CallOp
      // Each wrapper decides if it should write based on address
      auto methodSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_write");
      for (size_t i = 0; i < entryInfo.numBanks; ++i) {
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), wrapperInstances[i]->getName());
        bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{}, mlir::ValueRange{addr.getValue(), data.getValue()},
          calleeSymbol, methodSymbol, nullptr, nullptr);
      }

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    burstWrite->finalize();

    // Expose per-bank direct read/write methods that bypass burst translation.
    auto bankAddrType = UIntType::get(builder.getContext(), entryInfo.addrWidth);
    auto bankDataType = UIntType::get(builder.getContext(), entryInfo.dataWidth);

    for (size_t bankIdx = 0; bankIdx < entryInfo.numBanks; ++bankIdx) {
      std::string bankReadName = "bank_read_" + std::to_string(bankIdx);
      std::string bankWriteName = "bank_write_" + std::to_string(bankIdx);

      auto *bankReadMethod = entryModule->addMethod(
        bankReadName,
        {{"addr", bankAddrType}},
        {bankDataType}
      );

      bankReadMethod->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
        auto trueConst = UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
        guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
      });

      bankReadMethod->body([&, bankIdx](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
        auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), wrapperInstances[bankIdx]->getName());
        auto methodSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "bank_read");
        auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
          loc,
          mlir::TypeRange{bankDataType},
          mlir::ValueRange{addr.getValue()},
          calleeSymbol,
          methodSymbol,
          bodyBuilder.getArrayAttr({}),
          bodyBuilder.getArrayAttr({})
        );
        bodyBuilder.create<circt::cmt2::ReturnOp>(loc, callOp.getResult(0));
      });

      bankReadMethod->finalize();

      auto *bankWriteMethod = entryModule->addMethod(
        bankWriteName,
        {{"addr", bankAddrType}, {"data", bankDataType}},
        {}
      );

      bankWriteMethod->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
        auto trueConst = UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
        guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
      });

      bankWriteMethod->body([&, bankIdx](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
        auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());
        auto data = Signal(args[1], &bodyBuilder, entryModule->getLoc());
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), wrapperInstances[bankIdx]->getName());
        auto methodSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "bank_write");
        bodyBuilder.create<circt::cmt2::CallOp>(
          loc,
          mlir::TypeRange{},
          mlir::ValueRange{addr.getValue(), data.getValue()},
          calleeSymbol,
          methodSymbol,
          bodyBuilder.getArrayAttr({}),
          bodyBuilder.getArrayAttr({})
        );
        bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
      });

      bankWriteMethod->finalize();
    }

    return entryModule;
  }

  // ============================================================================
  //
  //                    MEMORY POOL (CONTAIN MEMORY ENTRY)
  //
  // ============================================================================

  /// Generate burst access logic (address decoding and bank selection)
  void generateBurstAccessLogic(Module *poolModule,
                                const llvm::SmallVector<MemoryEntryInfo> &memEntryInfos,
                                Circuit &circuit,
                                Clock clk, Reset rst) {

    auto &builder = poolModule->getBuilder();
    auto loc = poolModule->getLoc();
    auto u64Type = UIntType::get(builder.getContext(), 64);

    // TODO: refactor ----------------------------------------------------------------------------------------------------------------------------------
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

      auto *entryInst = poolModule->addInstance(instanceName, entryMod,
                                                {clk.getValue(), rst.getValue()});
      entryInstances.push_back(entryInst);
    }

    // Create top-level burst_read method that dispatches to correct memory entry
    auto *topBurstRead = poolModule->addMethod(
      "burst_read",
      {{"addr", u64Type}},
      {u64Type}
    );

    topBurstRead->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    topBurstRead->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, loc);
      auto resultInit = UInt::constant(0, 64, bodyBuilder, loc);
      mlir::Value result = resultInit.getValue();

      for (size_t i = 0; i < memEntryInfos.size(); ++i) {
        const auto &entryInfo = memEntryInfos[i];
        auto *entryInst = entryInstances[i];

        uint32_t entryStart = entryInfo.baseAddress;
        uint32_t entryEnd = entryStart + entryInfo.bankSize;

        // Calculate relative address for this entry
        auto startConst = UInt::constant(entryStart, 64, bodyBuilder, loc);
        auto endConst = UInt::constant(entryEnd, 64, bodyBuilder, loc);

        auto relAddr = (addr - startConst).bits(63, 0);
        auto inRange = (addr >= startConst) & (addr < endConst);

        // Call submodule's burst_read (manual CallOp needed for methods with return values)
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), entryInst->getName());
        auto methodSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_read");
        auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{u64Type}, mlir::ValueRange{relAddr.getValue()},
          calleeSymbol, methodSymbol, nullptr, nullptr);
        if (callOp.getNumResults() == 0) {
          bodyBuilder.getBlock()->getParentOp()->emitError()
              << "CallOp for burst_read returned no results (instance '"
              << entryInst->getName() << "')";
        }
        auto entryResult = Signal(callOp.getResult(0), &bodyBuilder, loc);

        // Use mux to conditionally include this result
        auto zeroData = UInt::constant(0, 64, bodyBuilder, loc);
        auto selectedData = inRange.mux(entryResult, zeroData);

        result = (Signal(result, &bodyBuilder, loc) | selectedData).getValue();
      }

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, result);
    });

    topBurstRead->finalize();

    // Create top-level burst_write method that dispatches to correct memory entry
    auto *topBurstWrite = poolModule->addMethod(
      "burst_write",
      {{"addr", u64Type}, {"data", u64Type}},
      {}  // No return value
    );

    topBurstWrite->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    topBurstWrite->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, poolModule->getLoc());
      auto data = Signal(args[1], &bodyBuilder, poolModule->getLoc());

      // Dispatch to all memory entries based on address range
      for (size_t i = 0; i < memEntryInfos.size(); ++i) {
        const auto &entryInfo = memEntryInfos[i];
        auto *entryInst = entryInstances[i];

        uint32_t entryStart = entryInfo.baseAddress;

        // Calculate relative address for this entry
        auto startConst = UInt::constant(entryStart, 64, bodyBuilder, poolModule->getLoc());
        auto relAddr = (addr - startConst).bits(63, 0);

        // Call submodule's burst_write
        // Note: We call all submodules, but only the one with matching address range will write
        // (the submodule's internal logic handles address decoding)
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), entryInst->getName());
        auto methodSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_write");
        bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{}, mlir::ValueRange{relAddr.getValue(), data.getValue()},
          calleeSymbol, methodSymbol, nullptr, nullptr);
      }

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    topBurstWrite->finalize();

    // Expose direct per-bank read/write methods at the top level (no burst translation).
    for (size_t entryIdx = 0; entryIdx < memEntryInfos.size(); ++entryIdx) {
      const auto &entryInfo = memEntryInfos[entryIdx];
      auto bankAddrType = UIntType::get(builder.getContext(), entryInfo.addrWidth);
      auto bankDataType = UIntType::get(builder.getContext(), entryInfo.dataWidth);

      for (size_t bankIdx = 0; bankIdx < entryInfo.numBanks; ++bankIdx) {
        std::string topReadName = entryInfo.name + "_bank" + std::to_string(bankIdx) + "_read";
        std::string topWriteName = entryInfo.name + "_bank" + std::to_string(bankIdx) + "_write";
        std::string entryReadName = "bank_read_" + std::to_string(bankIdx);
        std::string entryWriteName = "bank_write_" + std::to_string(bankIdx);

        auto *topBankRead = poolModule->addMethod(
          topReadName,
          {{"addr", bankAddrType}},
          {bankDataType}
        );

        topBankRead->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
          auto trueConst = UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
          guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
        });

        topBankRead->body([&, entryIdx, entryReadName](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
          auto addr = Signal(args[0], &bodyBuilder, poolModule->getLoc());
          auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), entryInstances[entryIdx]->getName());
          auto methodSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), entryReadName);
          auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
            loc,
            mlir::TypeRange{bankDataType},
            mlir::ValueRange{addr.getValue()},
            calleeSymbol,
            methodSymbol,
            bodyBuilder.getArrayAttr({}),
            bodyBuilder.getArrayAttr({})
          );
          bodyBuilder.create<circt::cmt2::ReturnOp>(loc, callOp.getResult(0));
        });

        topBankRead->finalize();

        auto *topBankWrite = poolModule->addMethod(
          topWriteName,
          {{"addr", bankAddrType}, {"data", bankDataType}},
          {}
        );

        topBankWrite->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
          auto trueConst = UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
          guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
        });

        topBankWrite->body([&, entryIdx, entryWriteName](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
          auto addr = Signal(args[0], &bodyBuilder, poolModule->getLoc());
          auto data = Signal(args[1], &bodyBuilder, poolModule->getLoc());
          auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), entryInstances[entryIdx]->getName());
          auto methodSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), entryWriteName);
          bodyBuilder.create<circt::cmt2::CallOp>(
            loc,
            mlir::TypeRange{},
            mlir::ValueRange{addr.getValue(), data.getValue()},
            calleeSymbol,
            methodSymbol,
            bodyBuilder.getArrayAttr({}),
            bodyBuilder.getArrayAttr({})
          );
          bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
        });

        topBankWrite->finalize();
      }
    }

  }

  /// Generate the CMT2 memory pool module
  Module *generateMemoryPool(Circuit &circuit, ModuleOp moduleOp, aps::MemoryMapOp memoryMapOp) {
    llvm::outs() << "DEBUG: generateMemoryPool() started\n";
    MLIRContext *context = moduleOp.getContext();
    OpBuilder builder(context);

    // Collect all memory entries
    llvm::SmallVector<aps::MemEntryOp> memEntries;
    llvm::outs() << "DEBUG: About to collect memory entries\n";

    if (memoryMapOp) {
      llvm::outs() << "DEBUG: Memory map region has " << memoryMapOp.getRegion().getBlocks().size() << " blocks\n";

      // Safely iterate through operations
      for (auto &block : memoryMapOp.getRegion()) {
        for (auto &op : block) {
          if (auto entry = dyn_cast<aps::MemEntryOp>(op)) {
            memEntries.push_back(entry);
            llvm::outs() << "DEBUG: Found memory entry: " << entry.getName() << "\n";
          }
        }
      }

      llvm::outs() << "DEBUG: Collected " << memEntries.size() << " memory entries\n";
    }

    if (memEntries.empty()) {
      llvm::outs() << "DEBUG: No memory entries found, will generate rules without memory\n";
    }

    // Generate memory pool module
    llvm::outs() << "DEBUG: Creating ScratchpadMemoryPool module\n";
    auto *poolModule = circuit.addModule("ScratchpadMemoryPool");

    // Add clock and reset
    llvm::outs() << "DEBUG: Adding clock and reset arguments\n";
    Clock clk = poolModule->addClockArgument("clk");
    Reset rst = poolModule->addResetArgument("rst");

    // Store instances for each memory entry (for address decoding)
    llvm::SmallVector<MemoryEntryInfo> memEntryInfos;

    // For each memory entry, create SRAM banks (only if entries exist)
    for (auto memEntry : memEntries) {
      auto bankSymbols = memEntry.getBankSymbols();
      uint32_t numBanks = memEntry.getNumBanks();
      uint32_t baseAddress = memEntry.getBaseAddress();
      uint32_t bankSize = memEntry.getBankSize();
      uint32_t cyclic = memEntry.getCyclic();

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
      auto globalOp = getGlobalMemRef(memEntry.getOperation(), firstBankSymAttr.getValue());
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

    return poolModule;
  }

  /// Convert MLIR type to FIRRTL type
  mlir::Type toFirrtlType(Type type, MLIRContext *ctx) {
    if (auto intType = dyn_cast<mlir::IntegerType>(type)) {
      if (intType.isUnsigned())
        return circt::firrtl::UIntType::get(ctx, intType.getWidth());
      if (intType.isSigned())
        return circt::firrtl::SIntType::get(ctx, intType.getWidth());
      return circt::firrtl::UIntType::get(ctx, intType.getWidth());
    }
    return {};
  }

  // ============================================================================
  //
  //                            MAIN LOGIC GENERATION
  //
  // ============================================================================

  /// Generate rule-based main module for TOR functions
  std::pair<Module*, Instance*> generateRuleBasedMainModule(ModuleOp moduleOp, Circuit &circuit,
                                   Module *poolModule) {
    // Create main module in the same circuit
    auto *mainModule = circuit.addModule("main");

    // Find all TOR functions that need rule generation
    llvm::SmallVector<tor::FuncOp, 4> torFuncs;
    moduleOp.walk([&](tor::FuncOp funcOp) {
      if (funcOp.getName() == "main") {  // Focus on main function for now
        torFuncs.push_back(funcOp);
      }
    });

    // Add clock and reset arguments
    auto mainClk = mainModule->addClockArgument("clk");
    auto mainRst = mainModule->addResetArgument("rst");

    // Add scratchpad pool instance - use the pool module we created earlier
    auto *poolInstance = mainModule->addInstance("scratchpad_pool",
                                                 poolModule, {mainClk.getValue(), mainRst.getValue()});

    // For each TOR function, generate rules
    for (auto funcOp : torFuncs) {
      generateRulesForFunction(mainModule, funcOp, poolInstance, circuit);
    }

    return {mainModule, poolInstance};
  }

  /// Add burst read/write methods to main module
  void addBurstMethodsToMainModule(Module *mainModule, Instance *poolInstance) {
    auto &builder = mainModule->getBuilder();
    auto *context = builder.getContext();

    // Add burst read method
    auto *burstReadMethod = mainModule->addMethod("burst_read",
        {{"addr", circt::firrtl::UIntType::get(context, 64)}},
        {circt::firrtl::UIntType::get(context, 64)});

    burstReadMethod->guard([](mlir::OpBuilder &b, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto loc = b.getUnknownLoc();
      auto one = b.create<circt::firrtl::ConstantOp>(
          loc, circt::firrtl::UIntType::get(b.getContext(), 1),
          llvm::APInt(1, 1));
      b.create<circt::cmt2::ReturnOp>(loc, mlir::ValueRange{one});
    });

    burstReadMethod->body([](mlir::OpBuilder &b, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto loc = b.getUnknownLoc();
      auto addrArg = args[0];

      // Call the scratchpad pool burst_read method
      auto result = b.create<circt::cmt2::CallOp>(
          loc, circt::firrtl::UIntType::get(b.getContext(), 64),
          mlir::ValueRange{addrArg},
          mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
          mlir::SymbolRefAttr::get(b.getContext(), "burst_read"),
          mlir::ArrayAttr(), mlir::ArrayAttr());

      b.create<circt::cmt2::ReturnOp>(loc, result.getResult(0));
    });
    burstReadMethod->finalize();

    // Add burst write method
    auto *burstWriteMethod = mainModule->addMethod("burst_write",
        {{"addr", circt::firrtl::UIntType::get(context, 64)},
         {"data", circt::firrtl::UIntType::get(context, 64)}},
        {});

    burstWriteMethod->guard([](mlir::OpBuilder &b, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto loc = b.getUnknownLoc();
      auto one = b.create<circt::firrtl::ConstantOp>(
          loc, circt::firrtl::UIntType::get(b.getContext(), 1),
          llvm::APInt(1, 1));
      b.create<circt::cmt2::ReturnOp>(loc, mlir::ValueRange{one});
    });

    burstWriteMethod->body([](mlir::OpBuilder &b, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto loc = b.getUnknownLoc();
      auto addrArg = args[0];
      auto dataArg = args[1];

      // Call the scratchpad pool burst_write method
      b.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{},
          mlir::ValueRange{addrArg, dataArg},
          mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
          mlir::SymbolRefAttr::get(b.getContext(), "burst_write"),
          mlir::ArrayAttr(), mlir::ArrayAttr());

      b.create<circt::cmt2::ReturnOp>(loc);
    });
    burstWriteMethod->finalize();

  }

  /// Generate rules for a specific TOR function - proper implementation from rulegenpass.cpp
  void generateRulesForFunction(Module *mainModule, tor::FuncOp funcOp,
                               Instance *poolInstance, Circuit &circuit) {
    MLIRContext *ctx = mainModule->getBuilder().getContext();

    // Collect operations per time slot.
    llvm::DenseMap<int64_t, SlotInfo> slotMap;
    SmallVector<int64_t, 8> slotOrder;

    auto insertOpIntoSlot = [&](Operation *op, int64_t slot) {
      auto &info = slotMap[slot];
      info.ops.push_back(op);
    };

    for (Operation &op : funcOp.getBody().getOps()) {
      if (isa<tor::TimeGraphOp>(op) || isa<tor::ReturnOp>(op))
        continue;

      if (auto startAttr = op.getAttrOfType<IntegerAttr>("starttime")) {
        int64_t slot = startAttr.getInt();
        insertOpIntoSlot(&op, slot);
        continue;
      }

      if (isa<arith::ConstantOp>(op))
        continue;

      op.emitError("missing required 'starttime' attribute for rule generation");
      return;
    }

    // Populate sorted slot order and validate supported ops.
    for (auto &kv : slotMap)
      slotOrder.push_back(kv.first);
    llvm::sort(slotOrder);

    for (int64_t slot : slotOrder) {
      for (Operation *op : slotMap[slot].ops) {
        if (isa<arith::ConstantOp, memref::GetGlobalOp>(op))
          continue;
        if (isa<tor::AddIOp, tor::SubIOp, tor::MulIOp, aps::MemLoad>(op))
          continue;
        op->emitError("unsupported operation for Target A rule generation");
        return;
      }
    }

    // Analyze cross-slot value uses.
    llvm::DenseMap<mlir::Value, SmallVector<CrossUseInfo *>> producerEdges;
    llvm::DenseMap<Operation *, SmallVector<CrossUseInfo *>> consumerEdges;
    llvm::DenseMap<std::pair<int64_t, int64_t>, unsigned> edgeCounts;
    SmallVector<std::unique_ptr<CrossUseInfo>, 8> crossUseStorage;

    // TOR function arguments are currently unsupported; ensure they are unused
    for (auto arg : funcOp.getArguments()) {
      if (!arg.use_empty()) {
        funcOp.emitError(
            "TOR function arguments are not supported in Target A lowering");
        return;
      }
    }

    auto getSlotForOp = [&](Operation *op) -> std::optional<int64_t> {
      if (auto attr = op->getAttrOfType<IntegerAttr>("starttime"))
        return attr.getInt();
      return {};
    };

    for (int64_t slot : slotOrder) {
      for (Operation *op : slotMap[slot].ops) {
        if (isa<arith::ConstantOp>(op))
          continue;
        for (mlir::Value res : op->getResults()) {
          if (!isa<mlir::IntegerType>(res.getType()))
            continue;

          for (OpOperand &use : res.getUses()) {
            Operation *user = use.getOwner();
            auto maybeConsumerSlot = getSlotForOp(user);
            if (!maybeConsumerSlot)
              continue;
            int64_t consumerSlot = *maybeConsumerSlot;
            if (consumerSlot <= slot)
              continue;

            auto firType = toFirrtlType(res.getType(), ctx);
            if (!firType) {
              op->emitError("type is unsupported for rule lowering");
              return;
            }

            auto key = std::make_pair(slot, consumerSlot);
            unsigned count = edgeCounts[key]++;

            auto info = std::make_unique<CrossUseInfo>();
            info->producerValue = res;
            info->producerSlot = slot;
            info->consumerOp = user;
            info->operandIndex = use.getOperandNumber();
            info->consumerSlot = consumerSlot;
            info->firType = firType;
            info->instanceName = "fifo_s" + std::to_string(slot) + "_s" +
                                 std::to_string(consumerSlot);
            if (count > 0)
              info->instanceName += "_" + std::to_string(count);

            producerEdges[res].push_back(info.get());
            consumerEdges[user].push_back(info.get());
            crossUseStorage.push_back(std::move(info));
          }
        }
      }
    }

    // Instantiate wire modules for cross-slot edges.
    for (auto &infoPtr : crossUseStorage) {
      auto *info = infoPtr.get();
      int64_t width = cast<circt::firrtl::UIntType>(info->firType).getWidthOrSentinel();
      if (width < 0)
        width = 1;
      auto *wireMod = STLLibrary::createWireModule(width, circuit);;
      info->wireInstance = mainModule->addInstance(info->instanceName, wireMod, {});
    }

    // Helper functions for value manipulation using Signal abstraction
    auto makeConstant = [&](mlir::OpBuilder &builder, Location opLoc,
                            uint64_t value, unsigned width) -> mlir::Value {
      auto constant = UInt::constant(value, width, builder, opLoc);
      return constant.getValue();
    };

    auto ensureUIntWidth = [&](mlir::OpBuilder &builder, Location valueLoc,
                               mlir::Value value, unsigned targetWidth,
                               Operation *sourceOp) -> FailureOr<mlir::Value> {
      if (targetWidth == 0)
        targetWidth = 1;

      auto type = cast<circt::firrtl::UIntType>(value.getType());
      if (!type) {
        sourceOp->emitError("expected FIRRTL UInt type");
        return failure();
      }

      int64_t currentWidth = type.getWidthOrSentinel();
      if (currentWidth < 0) {
        sourceOp->emitError("requires statically known bitwidth");
        return failure();
      }

      if (static_cast<unsigned>(currentWidth) == targetWidth)
        return value;

      // Use Signal abstraction for width manipulation
      auto signal = Signal(value, &builder, valueLoc);
      if (static_cast<unsigned>(currentWidth) < targetWidth) {
        // Pad to target width
        auto paddedSignal = signal.pad(targetWidth);
        return paddedSignal.getValue();
      } else {
        // Truncate to target width
        auto truncatedSignal = signal.bits(targetWidth - 1, 0);
        return truncatedSignal.getValue();
      }
    };

    // Helper to materialise values inside a rule.
    auto getValueInRule = [&](mlir::Value v, Operation *currentOp,
                              unsigned operandIndex,
                              mlir::OpBuilder &builder,
                              DenseMap<mlir::Value, mlir::Value> &localMap,
                              Location opLoc) -> FailureOr<mlir::Value> {
      if (auto it = localMap.find(v); it != localMap.end())
        return it->second;

      if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        auto intAttr = mlir::cast<IntegerAttr>(constOp.getValueAttr());
        unsigned width = mlir::cast<IntegerType>(intAttr.getType()).getWidth();
        auto constant = makeConstant(builder, opLoc, intAttr.getValue().getZExtValue(), width);
        localMap[v] = constant;
        return constant;
      }

      if (auto globalOp = v.getDefiningOp<memref::GetGlobalOp>()) {
        // Global symbols are handled separately via symbol resolution.
        return mlir::Value{};
      }

      // Check if this is a cross-slot use.
      auto it = producerEdges.find(v);
      if (it != producerEdges.end()) {
        for (CrossUseInfo *info : it->second) {
          if (info->consumerOp == currentOp && info->operandIndex == operandIndex) {
            auto result = info->wireInstance->callValue("read", builder);
            assert(result.size() == 1);
            localMap[v] = result[0];
            return result[0];
          }
        }
      }

      currentOp->emitError("value is not available in this rule");
      return failure();
    };

    // Generate rules for each time slot.
    llvm::DenseMap<mlir::Value, mlir::Value> localMap;
    llvm::SmallVector<std::pair<std::string, std::string>, 4> precedencePairs;

    for (int64_t slot : slotOrder) {
      auto *rule = mainModule->addRule("rule_s" + std::to_string(slot));

      // Guard: always ready (constant 1)
      rule->guard([](mlir::OpBuilder &b) {
        auto loc = b.getUnknownLoc();
        auto one = UInt::constant(1, 1, b, loc);
        b.create<circt::cmt2::ReturnOp>(loc, mlir::ValueRange{one.getValue()});
      });

      // Body: implement the operations for this time slot
      rule->body([&](mlir::OpBuilder &b) {
        auto loc = b.getUnknownLoc();
        localMap.clear();

        for (Operation *op : slotMap[slot].ops) {
          if (auto addOp = dyn_cast<tor::AddIOp>(op)) {
            auto lhs = getValueInRule(addOp.getLhs(), op, 0, b, localMap, loc);
            auto rhs = getValueInRule(addOp.getRhs(), op, 1, b, localMap, loc);
            if (failed(lhs) || failed(rhs))
              return;

            // Use Signal for arithmetic operations
            auto lhsSignal = Signal(*lhs, &b, loc);
            auto rhsSignal = Signal(*rhs, &b, loc);

            // Determine result width and extend if needed
            auto lhsWidth = cast<circt::firrtl::UIntType>((*lhs).getType()).getWidthOrSentinel();
            auto rhsWidth = cast<circt::firrtl::UIntType>((*rhs).getType()).getWidthOrSentinel();
            auto resultWidth = std::max(lhsWidth, rhsWidth);

            auto lhsExtended = ensureUIntWidth(b, loc, *lhs, resultWidth, op);
            auto rhsExtended = ensureUIntWidth(b, loc, *rhs, resultWidth, op);
            if (failed(lhsExtended) || failed(rhsExtended))
              return;

            auto sumSignal = Signal(*lhsExtended, &b, loc) + Signal(*rhsExtended, &b, loc);
            localMap[addOp.getResult()] = sumSignal.getValue();
          }
          else if (auto subOp = dyn_cast<tor::SubIOp>(op)) {
            auto lhs = getValueInRule(subOp.getLhs(), op, 0, b, localMap, loc);
            auto rhs = getValueInRule(subOp.getRhs(), op, 1, b, localMap, loc);
            if (failed(lhs) || failed(rhs))
              return;

            // Use Signal for arithmetic operations
            auto lhsWidth = cast<circt::firrtl::UIntType>((*lhs).getType()).getWidthOrSentinel();
            auto rhsWidth = cast<circt::firrtl::UIntType>((*rhs).getType()).getWidthOrSentinel();
            auto resultWidth = std::max(lhsWidth, rhsWidth);

            auto lhsExtended = ensureUIntWidth(b, loc, *lhs, resultWidth, op);
            auto rhsExtended = ensureUIntWidth(b, loc, *rhs, resultWidth, op);
            if (failed(lhsExtended) || failed(rhsExtended))
              return;

            auto diffSignal = Signal(*lhsExtended, &b, loc) - Signal(*rhsExtended, &b, loc);
            localMap[subOp.getResult()] = diffSignal.getValue();
          }
          else if (auto mulOp = dyn_cast<tor::MulIOp>(op)) {
            auto lhs = getValueInRule(mulOp.getLhs(), op, 0, b, localMap, loc);
            auto rhs = getValueInRule(mulOp.getRhs(), op, 1, b, localMap, loc);
            if (failed(lhs) || failed(rhs))
              return;

            // Use Signal for arithmetic operations
            auto lhsWidth = cast<circt::firrtl::UIntType>((*lhs).getType()).getWidthOrSentinel();
            auto rhsWidth = cast<circt::firrtl::UIntType>((*rhs).getType()).getWidthOrSentinel();
            auto resultWidth = lhsWidth + rhsWidth;

            auto lhsExtended = ensureUIntWidth(b, loc, *lhs, resultWidth, op);
            auto rhsExtended = ensureUIntWidth(b, loc, *rhs, resultWidth, op);
            if (failed(lhsExtended) || failed(rhsExtended))
              return;

            auto prodSignal = Signal(*lhsExtended, &b, loc) * Signal(*rhsExtended, &b, loc);
            localMap[mulOp.getResult()] = prodSignal.getValue();
          }
          else if (auto memLoad = dyn_cast<aps::MemLoad>(op)) {
            // Handle memory load operations
            auto addr = getValueInRule(memLoad.getIndices()[0], op, 0, b, localMap, loc);
            if (failed(addr))
              return;


            // Call the scratchpad pool bank read method
            auto callResult = b.create<circt::cmt2::CallOp>(
                loc, circt::firrtl::UIntType::get(b.getContext(), 64),
                mlir::ValueRange{*addr},
                mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
                mlir::SymbolRefAttr::get(b.getContext(), "scratch_bank0_read"),
                mlir::ArrayAttr(), mlir::ArrayAttr());

            localMap[memLoad.getResult()] = callResult.getResult(0);
          }
          else if (auto consumerEdgeIt = consumerEdges.find(op);
                     consumerEdgeIt != consumerEdges.end()) {
            // Handle cross-slot value writes
            for (CrossUseInfo *info : consumerEdgeIt->second) {
              auto value = getValueInRule(info->producerValue, op, info->operandIndex, b, localMap, loc);
              if (failed(value))
                return;
              info->wireInstance->callMethod("write", {*value}, b);
            }
          }
        }

        b.create<circt::cmt2::ReturnOp>(loc);
      });

      rule->finalize();

      // Add precedence constraints - find previous slot in order
      if (!slotOrder.empty() && slot != slotOrder[0]) {
        auto it = std::find(slotOrder.begin(), slotOrder.end(), slot);
        if (it != slotOrder.begin()) {
          int64_t prevSlot = *(it - 1);
          std::string from = "@rule_s" + std::to_string(prevSlot);
          std::string to = "@rule_s" + std::to_string(slot);
          precedencePairs.emplace_back(from, to);
        }
      }
    }

    if (!precedencePairs.empty())
      mainModule->setPrecedence(precedencePairs);
  }
};
} // namespace mlir

std::unique_ptr<mlir::Pass> mlir::createAPSToCMT2GenPass() {
  return std::make_unique<mlir::APSToCMT2GenPass>();
}
