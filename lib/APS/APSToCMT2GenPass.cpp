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
#include "circt/Dialect/Cmt2/ECMT2/Interface.h"
#include "circt/Dialect/Cmt2/ECMT2/Module.h"
#include "circt/Dialect/Cmt2/ECMT2/ModuleLibrary.h"
#include "circt/Dialect/Cmt2/ECMT2/STLLibrary.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "circt/Dialect/Cmt2/ECMT2/SignalHelpers.h"
#include "circt/Dialect/Cmt2/Cmt2Dialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "mlir-c/Support.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <bit>

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

struct CrossSlotFIFO {
  mlir::Value producerValue;      // The SSA value being communicated
  int64_t producerSlot = 0;       // Source time slot
  int64_t consumerSlot = 0;       // Destination time slot
  std::string instanceName;       // FIFO instance name
  mlir::Type firType;            // FIRRTL type for the value
  Instance *fifoInstance = nullptr; // FIFO module instance
  llvm::SmallVector<std::pair<Operation*, unsigned>> consumers; // (consumerOp, operandIndex) pairs
};

  uint32_t roundUpToPowerOf2(uint32_t value) {
      if (value <= 1) return 1;
      value--;
      value |= value >> 1;
      value |= value >> 2;
      value |= value >> 4;
      value |= value >> 8;
      value |= value >> 16;
      value++;
      return value;
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

  /// Result struct for generateMemoryPool containing module and memory entry map
  struct MemoryPoolResult {
    Module* poolModule;
    llvm::DenseMap<llvm::StringRef, MemoryEntryInfo> memEntryMap;
  };

struct APSToCMT2GenPass : public PassWrapper<APSToCMT2GenPass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(APSToCMT2GenPass)

  // Memory entry map for fast lookup by name
  llvm::DenseMap<llvm::StringRef, MemoryEntryInfo> memEntryMap;

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

    MLIRContext *context = moduleOp.getContext();
    Circuit circuit("MemoryPool", *context);

    addBurstMemoryInterface(circuit);
    auto memoryPoolResult = generateMemoryPool(circuit, moduleOp, memoryMapOp);
    Module *poolModule = memoryPoolResult.poolModule;
    memEntryMap = std::move(memoryPoolResult.memEntryMap);

    // Generate RoCC adapter module
    // TODO: Extract opcodes from the TOR functions or pass as parameter
    llvm::SmallVector<uint32_t, 4> opcodes = {0b0001011, 0b0101011}; // Example opcodes
    auto *roccAdapterModule = generateRoCCAdapter(circuit, opcodes);

    // // Generate memory translator module
    auto *memoryAdapterModule = generateMemoryAdapter(circuit);
    
    // Generate rule-based main module for TOR functions
    auto [mainModule, poolInstance] = generateRuleBasedMainModule(moduleOp, circuit, poolModule, roccAdapterModule, memoryAdapterModule);

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

  void addBurstMemoryInterface(Circuit &circuit) {
    auto &context = circuit.getContext();
    auto *burstMemoryInterface = circuit.addInterface("BurstDMAController");
    auto u32Type = UIntType::get(&context, 32);
    auto u4Type = UIntType::get(&context, 4);
    auto u1Type = UIntType::get(&context, 1);
    burstMemoryInterface->addMethod(
      "cpu_to_isax",
      {
        {"cpu_addr", u32Type},
        {"isax_addr", u32Type},
        {"length", u4Type}
      },
      {}
    );
    burstMemoryInterface->addMethod(
      "isax_to_cpu",
      {
        {"cpu_addr", u32Type},
        {"isax_addr", u32Type},
        {"length", u4Type}
      },
      {}
    );
    burstMemoryInterface->addValue(
      // This will not ready if burst engine is running
      // So the main operation can be stucked
      "poll_for_idle",
      {}, 
      {TypeAttr::get(u1Type)}
    );
  }

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

      auto elementSizeConst = UInt::constant(entryInfo.dataWidth / 8, 64, bodyBuilder, wrapper->getLoc());  // Element size in bytes
      auto numBanksConst = UInt::constant(entryInfo.numBanks, 64, bodyBuilder, wrapper->getLoc());
      auto myBankConst = UInt::constant(bankIdx, 64, bodyBuilder, wrapper->getLoc());
      auto elementsPerBurstConst = UInt::constant(elementsPerBurst, 64, bodyBuilder, wrapper->getLoc());

      // element_idx = addr / element_size
      auto elementIdx = addr / elementSizeConst;
      // start_bank_idx = element_idx % num_banks
      auto startBankIdx = elementIdx % numBanksConst;

      // Check if this bank participates: participatesInBurst = (position < elements_per_burst)
      // position = (my_bank - start_bank_idx + num_banks) % num_banks
      auto position = (myBankConst - startBankIdx + numBanksConst) % numBanksConst;
      auto isMine = position < elementsPerBurstConst;

      // Calculate local address: my_element_idx = element_idx + position; local_addr = my_element_idx / num_banks
      auto myElementIdx = elementIdx + position;
      auto localAddr = myElementIdx / numBanksConst;
      auto localAddrTrunc = localAddr.bits(entryInfo.addrWidth - 1, 0);

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
      auto trueConst = UInt::constant(1, 1, guardBuilder, loc);
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    burstWrite->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());
      auto data = Signal(args[1], &bodyBuilder, wrapper->getLoc());

      // Helper: Create constants using Signal
      auto elementSizeConst = UInt::constant(entryInfo.dataWidth / 8, 64, bodyBuilder, wrapper->getLoc());  // Element size in bytes
      auto numBanksConst = UInt::constant(entryInfo.numBanks, 64, bodyBuilder, wrapper->getLoc());
      auto myBankConst = UInt::constant(bankIdx, 64, bodyBuilder, wrapper->getLoc());
      auto elementsPerBurstConst = UInt::constant(elementsPerBurst, 64, bodyBuilder, wrapper->getLoc());

      // Helper 1: element_idx = addr / element_size
      auto elementIdx = addr / elementSizeConst;

      // Helper 2: start_bank_idx = element_idx % num_banks
      auto startBankIdx = elementIdx % numBanksConst;

      // Helper 3: Calculate position and check participation
      auto position = ((myBankConst - startBankIdx + numBanksConst) % numBanksConst);
      auto isMine = position < elementsPerBurstConst;

      // Helper 4: Calculate local address
      auto myElementIdx = elementIdx + position;
      auto localAddr = myElementIdx / numBanksConst;
      auto localAddrTrunc = localAddr.bits(entryInfo.addrWidth - 1, 0);

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

    // Step 1: Create all memory entry submodules
    // Save the insertion point first, since circuit.addModule() will change it
    auto savedIP = builder.saveInsertionPoint();

    llvm::SmallVector<Module*, 4> entryModules;
    for (const auto &entryInfo : memEntryInfos) {
      // Collect bank names
      llvm::SmallVector<std::string, 4> bankNames;
      for (size_t i = 0; i < entryInfo.numBanks; ++i) {
        bankNames.push_back(entryInfo.name + "_" + std::to_string(i));
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
        uint32_t entryEnd = entryStart + entryInfo.bankSize;

        // Calculate relative address for this entry
        auto startConst = UInt::constant(entryStart, 64, bodyBuilder, poolModule->getLoc());
        auto endConst = UInt::constant(entryEnd, 64, bodyBuilder, poolModule->getLoc());

        auto relAddr = (addr - startConst).bits(63, 0);
        auto inRange = (addr >= startConst) & (addr < endConst);

        // Only call submodule's burst_write if address is in range
        // Use If to conditionally execute the call
        If(inRange, [&](mlir::OpBuilder &thenBuilder) {
          auto calleeSymbol = mlir::FlatSymbolRefAttr::get(thenBuilder.getContext(), entryInst->getName());
          auto methodSymbol = mlir::FlatSymbolRefAttr::get(thenBuilder.getContext(), "burst_write");
          thenBuilder.create<circt::cmt2::CallOp>(
            loc, mlir::TypeRange{}, mlir::ValueRange{relAddr.getValue(), data.getValue()},
            calleeSymbol, methodSymbol, nullptr, nullptr);
        }, bodyBuilder, loc);
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
  MemoryPoolResult generateMemoryPool(Circuit &circuit, ModuleOp moduleOp, aps::MemoryMapOp memoryMapOp) {
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

    // Create memory entry map for fast lookup by name
    llvm::DenseMap<llvm::StringRef, MemoryEntryInfo> memEntryMap;

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

      // Add this memory entry info to the list and map
      memEntryInfos.push_back(entryInfo);
      memEntryMap[memEntry.getName()] = std::move(entryInfo);
    }

    // Now generate address decoding logic for burst access
    // This will create memory entry submodules with bank instances
    generateBurstAccessLogic(poolModule, memEntryInfos, circuit, clk, rst);

    return MemoryPoolResult{poolModule, std::move(memEntryMap)};
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
                                   Module *poolModule, Module *roccModule, Module *hellaMemModule) {
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
    auto *burstControllerItfcDecl = mainModule->defineInterfaceDecl("dma", "BurstDMAController");

    // Add scratchpad pool instance - use the pool module we created earlier
    auto *poolInstance = mainModule->addInstance("scratchpad_pool", poolModule, {mainClk.getValue(), mainRst.getValue()});
    auto *roccInstance = mainModule->addInstance("rocc_adapter", roccModule, {mainClk.getValue(), mainRst.getValue()});
    auto *hellaMemInstance = mainModule->addInstance("hellacache_adapter", hellaMemModule, {mainClk.getValue(), mainRst.getValue()});

    // For each TOR function, generate rules
    // Extract opcodes from the function or use defaults
    // For scalar.mlir, we need to determine the appropriate opcode
    uint32_t opcode = 0b0001011; // Default opcode for scalar operations
    for (auto funcOp : torFuncs) {
      generateRulesForFunction(mainModule, funcOp, poolInstance, roccInstance, hellaMemInstance, burstControllerItfcDecl, circuit, mainClk, mainRst, opcode);
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

  /// Generate Memory Translator module that bridges HellaCache interface with User Memory Protocol
  Module *generateMemoryAdapter(Circuit &circuit) {
    auto *translatorModule = circuit.addModule("MemoryTranslator");
    auto &builder = translatorModule->getBuilder();
    auto loc = translatorModule->getLoc();
    auto *context = builder.getContext();

    // Add clock and reset arguments
    Clock clk = translatorModule->addClockArgument("clk");
    Reset rst = translatorModule->addResetArgument("rst");

    // Define types for the protocols (using inline definitions to avoid unused variables)

    // Create bundle types following Rust patterns
    // UserMemoryCmd: {addr: u32, cmd: u1, size: u2, data: u32, mask: u4, tag: u8}
    auto userCmdBundleType = BundleType::get(context, {
      BundleType::BundleElement{builder.getStringAttr("addr"), false, UIntType::get(context, 32)},
      BundleType::BundleElement{builder.getStringAttr("cmd"), false, UIntType::get(context, 1)},
      BundleType::BundleElement{builder.getStringAttr("size"), false, UIntType::get(context, 2)},
      BundleType::BundleElement{builder.getStringAttr("data"), false, UIntType::get(context, 32)},
      BundleType::BundleElement{builder.getStringAttr("mask"), false, UIntType::get(context, 4)},
      BundleType::BundleElement{builder.getStringAttr("tag"), false, UIntType::get(context, 8)}
    });

    // HellaCacheCmd: {addr: u32, tag: u8, cmd: u5, size: u2, signed: u1, phys: u1, data: u32, mask: u4}
    auto hellaCmdBundleType = BundleType::get(context, {
      BundleType::BundleElement{builder.getStringAttr("addr"), false, UIntType::get(context, 32)},
      BundleType::BundleElement{builder.getStringAttr("tag"), false, UIntType::get(context, 8)},
      BundleType::BundleElement{builder.getStringAttr("cmd"), false, UIntType::get(context, 5)},
      BundleType::BundleElement{builder.getStringAttr("size"), false, UIntType::get(context, 2)},
      BundleType::BundleElement{builder.getStringAttr("signed"), false, UIntType::get(context, 1)},
      BundleType::BundleElement{builder.getStringAttr("phys"), false, UIntType::get(context, 1)},
      BundleType::BundleElement{builder.getStringAttr("data"), false, UIntType::get(context, 32)},
      BundleType::BundleElement{builder.getStringAttr("mask"), false, UIntType::get(context, 4)}
    });

    // HellaCacheResp: {data: u32, tag: u8, cmd: u5, size: u2, signed: u1}
    auto hellaRespBundleType = BundleType::get(context, {
      BundleType::BundleElement{builder.getStringAttr("data"), false, UIntType::get(context, 32)},
      BundleType::BundleElement{builder.getStringAttr("tag"), false, UIntType::get(context, 8)},
      BundleType::BundleElement{builder.getStringAttr("cmd"), false, UIntType::get(context, 5)},
      BundleType::BundleElement{builder.getStringAttr("size"), false, UIntType::get(context, 2)},
      BundleType::BundleElement{builder.getStringAttr("signed"), false, UIntType::get(context, 1)}
    });

    // UserMemoryResp: {data: u32, tag: u8}
    auto userRespBundleType = BundleType::get(context, {
      BundleType::BundleElement{builder.getStringAttr("data"), false, UIntType::get(context, 32)},
      BundleType::BundleElement{builder.getStringAttr("tag"), false, UIntType::get(context, 8)}
    });

    // Create external FIFO and register modules
    auto savedIP = builder.saveInsertionPoint();

    // Command queue (FIFO1Push for HellaCache commands - 82 bits total)
    auto hellaCmdFifoMod = STLLibrary::createFIFO1PushModule(82, circuit);

    // Response buffer registers (grouped by bitwidth for reuse)
    auto reg32Mod = STLLibrary::createRegModule(32, 0, circuit);  // 32-bit registers for data
    auto reg8Mod = STLLibrary::createRegModule(8, 0, circuit);  // 32-bit registers for tags
    auto reg1Mod = STLLibrary::createRegModule(1, 0, circuit);    // 1-bit registers for flags

    // Wire modules for logic (grouped by bitwidth)
    auto wire1Mod = STLLibrary::createWireModule(1, circuit);     // 1-bit wires for logic

    builder.restoreInsertionPoint(savedIP);

    // Create instances
    auto *hellaCmdFifo = translatorModule->addInstance("hella_cmd_fifo", hellaCmdFifoMod,
                                                       {clk.getValue(), rst.getValue()});

    // Slot 0 instances (registers don't need clock/reset)
    auto *slot0DataReg = translatorModule->addInstance("slot0_data_reg", reg32Mod, {});
    auto *slot0TagReg = translatorModule->addInstance("slot0_tag_reg", reg8Mod, {});
    auto *slot0TxdReg = translatorModule->addInstance("slot0_txd_reg", reg1Mod, {});
    auto *slot0RxdReg = translatorModule->addInstance("slot0_rxd_reg", reg1Mod, {});

    // Slot 1 instances (registers don't need clock/reset)
    auto *slot1DataReg = translatorModule->addInstance("slot1_data_reg", reg32Mod, {});
    auto *slot1TagReg = translatorModule->addInstance("slot1_tag_reg", reg8Mod, {});
    auto *slot1TxdReg = translatorModule->addInstance("slot1_txd_reg", reg1Mod, {});
    auto *slot1RxdReg = translatorModule->addInstance("slot1_rxd_reg", reg1Mod, {});

    // Control flag instances (registers don't need clock/reset)
    auto *newerSlotReg = translatorModule->addInstance("newer_slot_reg", reg1Mod, {});

    // Wire instances for logic
    auto *slot0CanCollectWire = translatorModule->addInstance("slot0_can_collect_wire", wire1Mod, {});
    auto *slot1CanCollectWire = translatorModule->addInstance("slot1_can_collect_wire", wire1Mod, {});

    // Method: cmd_from_user - receives user memory commands and translates to HellaCache format
    std::vector<std::pair<std::string, mlir::Type>> cmdFromUserArgs = {{"user_cmd", userCmdBundleType}};
    auto *cmdFromUser = translatorModule->addMethod("cmd_from_user", cmdFromUserArgs, {});

    cmdFromUser->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument>) {
      auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
    });

    cmdFromUser->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      Bundle userCmd(args[0], &bodyBuilder, loc);

      // Extract user command fields
      Signal userAddr = userCmd["addr"];
      Signal userTag = userCmd["tag"];
      Signal userCmdType = userCmd["cmd"];
      Signal userSize = userCmd["size"];
      Signal userData = userCmd["data"];
      Signal userMask = userCmd["mask"];

      // Translate to HellaCache format (following Rust logic)
      // cmd = user_cmd (0=read, 1=write) -> same, but 5-bit wide
      Signal hellaCmd = userCmdType.pad(5);
      // signed = false, phys = false (following Rust)
      Signal hellaSigned = UInt::constant(0, 1, bodyBuilder, loc);
      Signal hellaPhys = UInt::constant(0, 1, bodyBuilder, loc);

      // Pack the bundle fields into a 82-bit value for FIFO
      // Layout: data(32) + mask(4) + phys(1) + signed(1) + size(2) + cmd(5) + tag(8) + addr(32) = 85 bits (with padding)
      Signal packedHellaCmd = userData.cat(userMask).cat(hellaPhys).cat(hellaSigned)
                              .cat(userSize).cat(hellaCmd).cat(userTag).cat(userAddr);

      // Enqueue to HellaCache command FIFO
      hellaCmdFifo->callMethod("enq", {packedHellaCmd.getValue()}, bodyBuilder);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    cmdFromUser->finalize();

    // Rule: commit_cmd - processes HellaCache commands and manages response slots
    // This is equivalent to the Rust always! block for command processing
    auto *commitCmdRule = translatorModule->addRule("commit_cmd");

    commitCmdRule->guard([&](mlir::OpBuilder &guardBuilder) {
      auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
    });

    commitCmdRule->body([&](mlir::OpBuilder &bodyBuilder) {
      // Pop from HellaCache command queue
      auto cmdValues = hellaCmdFifo->callValue("deq", bodyBuilder);
      auto hellaCmd = Signal(cmdValues[0], &bodyBuilder, loc);

      // Extract fields from HellaCache command (assuming packed format)
      // Layout: data(32) + mask(4) + phys(1) + signed(1) + size(2) + cmd(5) + tag(8) + addr(32)
      Signal tagAddr = hellaCmd.bits(39, 32);     // bits 32-39: tag used as address for matching
      Signal cmdField = hellaCmd.bits(46, 42);     // bits 42-46: cmd (5-bit)

      // Check if this is a read command (cmd == 0)
      Signal cmdIsRead = cmdField == UInt::constant(0, 5, bodyBuilder, loc);

      // Read current newer_slot flag
      auto newerSlotValues = newerSlotReg->callValue("read", bodyBuilder);
      Signal newerSlot = Signal(newerSlotValues[0], &bodyBuilder, loc);

      // Command processing logic for read commands (following Rust pattern)
      If(cmdIsRead,
          [&](mlir::OpBuilder &builder) -> Signal {
            // Slot selection logic: if newer_slot == 1, use slot 0; else use slot 1
            auto slotLogic = If(newerSlot == UInt::constant(1, 1, builder, loc),
                [&](mlir::OpBuilder &innerBuilder) -> Signal {
                  // next slot is 0 - use slot 0 for tracking
                  slot0TxdReg->callMethod("write", {UInt::constant(1, 1, innerBuilder, loc).getValue()}, innerBuilder);
                  slot0TagReg->callMethod("write", {tagAddr.getValue()}, innerBuilder);
                  newerSlotReg->callMethod("write", {UInt::constant(0, 1, innerBuilder, loc).getValue()}, innerBuilder);
                  return UInt::constant(0, 1, innerBuilder, loc);
                },
                [&](mlir::OpBuilder &innerBuilder) -> Signal {
                  // next slot is 1 - use slot 1 for tracking
                  slot1TxdReg->callMethod("write", {UInt::constant(1, 1, innerBuilder, loc).getValue()}, innerBuilder);
                  slot1TagReg->callMethod("write", {tagAddr.getValue()}, innerBuilder);
                  newerSlotReg->callMethod("write", {UInt::constant(1, 1, innerBuilder, loc).getValue()}, innerBuilder);
                  return UInt::constant(0, 1, innerBuilder, loc);
                },
                builder, loc);
            return slotLogic;
          },
          [&](mlir::OpBuilder &builder) -> Signal {
            // For write commands or other operations, no slot management needed
            return UInt::constant(0, 1, builder, loc);
          },
          bodyBuilder, loc);

      // Note: In a complete implementation, we would also need to:
      // 1. Send the command to the actual HellaCache interface (hella_slave.cmd_to_bus)
      // 2. Handle write response processing if needed
      // This would require connecting to external HellaCache interface modules

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    commitCmdRule->finalize();

    // Rule: slot_0_can_collect_logic - computes when slot 0 can provide responses
    auto *slot0CanCollectLogicRule = translatorModule->addRule("slot_0_can_collect_logic");

    slot0CanCollectLogicRule->guard([&](mlir::OpBuilder &guardBuilder) {
      // This rule always fires (combinational logic)
      auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
    });

    slot0CanCollectLogicRule->body([&](mlir::OpBuilder &bodyBuilder) {
      // Read slot 0 flags
      auto slot0TxdValues = slot0TxdReg->callValue("read", bodyBuilder);
      auto slot0RxdValues = slot0RxdReg->callValue("read", bodyBuilder);
      Signal slot0Txd = Signal(slot0TxdValues[0], &bodyBuilder, loc);
      Signal slot0Rxd = Signal(slot0RxdValues[0], &bodyBuilder, loc);

      // Read slot 1 flags for ordering comparison
      auto slot1TxdValues = slot1TxdReg->callValue("read", bodyBuilder);
      auto newerSlotValues = newerSlotReg->callValue("read", bodyBuilder);
      Signal slot1Txd = Signal(slot1TxdValues[0], &bodyBuilder, loc);
      Signal newerSlot = Signal(newerSlotValues[0], &bodyBuilder, loc);

      // Compute slot0_ready: both transmitted and received flags set
      Signal slot0Ready = slot0Txd & slot0Rxd;

      // Compute is_earlier: slot 0 is earlier if slot 1 not transmitted OR
      // slot 1 transmitted AND newer_slot == 1 (slot 0 is the newer one)
      Signal slot1NotTx = slot1Txd == UInt::constant(0, 1, bodyBuilder, loc);
      Signal slot1TxAndNewer1 = slot1Txd & (newerSlot == UInt::constant(1, 1, bodyBuilder, loc));
      Signal isEarlier = slot1NotTx | slot1TxAndNewer1;

      // Compute can_collect: slot is ready AND is earlier
      Signal canCollect = slot0Ready & isEarlier;

      // Write to the wire
      slot0CanCollectWire->callMethod("write", {canCollect.getValue()}, bodyBuilder);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    slot0CanCollectLogicRule->finalize();

    // Rule: slot_1_can_collect_logic - computes when slot 1 can provide responses
    auto *slot1CanCollectLogicRule = translatorModule->addRule("slot_1_can_collect_logic");

    slot1CanCollectLogicRule->guard([&](mlir::OpBuilder &guardBuilder) {
      // This rule always fires (combinational logic)
      auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
    });

    slot1CanCollectLogicRule->body([&](mlir::OpBuilder &bodyBuilder) {
      // Read slot 1 flags
      auto slot1TxdValues = slot1TxdReg->callValue("read", bodyBuilder);
      auto slot1RxdValues = slot1RxdReg->callValue("read", bodyBuilder);
      Signal slot1Txd = Signal(slot1TxdValues[0], &bodyBuilder, loc);
      Signal slot1Rxd = Signal(slot1RxdValues[0], &bodyBuilder, loc);

      // Read slot 0 flags for ordering comparison
      auto slot0TxdValues = slot0TxdReg->callValue("read", bodyBuilder);
      auto newerSlotValues = newerSlotReg->callValue("read", bodyBuilder);
      Signal slot0Txd = Signal(slot0TxdValues[0], &bodyBuilder, loc);
      Signal newerSlot = Signal(newerSlotValues[0], &bodyBuilder, loc);

      // Compute slot1_ready: both transmitted and received flags set
      Signal slot1Ready = slot1Txd & slot1Rxd;

      // Compute is_earlier: slot 1 is earlier if slot 0 not transmitted OR
      // slot 0 transmitted AND newer_slot == 0 (slot 1 is the newer one)
      Signal slot0NotTx = slot0Txd == UInt::constant(0, 1, bodyBuilder, loc);
      Signal slot0TxAndNewer0 = slot0Txd & (newerSlot == UInt::constant(0, 1, bodyBuilder, loc));
      Signal isEarlier = slot0NotTx | slot0TxAndNewer0;

      // Compute can_collect: slot is ready AND is earlier
      Signal canCollect = slot1Ready & isEarlier;

      // Write to the wire
      slot1CanCollectWire->callMethod("write", {canCollect.getValue()}, bodyBuilder);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    slot1CanCollectLogicRule->finalize();

    // Method: resp_from_bus - receives responses from HellaCache and buffers them
    std::vector<std::pair<std::string, mlir::Type>> respFromBusArgs = {{"hella_resp", hellaRespBundleType}};
    auto *respFromBus = translatorModule->addMethod("resp_from_bus", respFromBusArgs, {});

    respFromBus->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument>) {
      auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
    });

    respFromBus->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      Bundle hellaResp(args[0], &bodyBuilder, loc);

      // Extract response fields
      Signal respData = hellaResp["data"];
      Signal respTag = hellaResp["tag"];
      Signal respCmd = hellaResp["cmd"];

      // Check if this is a read response (cmd == 0)
      Signal cmdIsRead = respCmd == UInt::constant(0, 5, bodyBuilder, loc);
      Signal tagAddr = respTag; // Use tag as address for matching

      // Read current newer_slot flag
      auto newerSlotValues = newerSlotReg->callValue("read", bodyBuilder);
      Signal newerSlot = Signal(newerSlotValues[0], &bodyBuilder, loc);

      // Conditional logic following Rust pattern
      auto result = If(cmdIsRead,
          [&](mlir::OpBuilder &builder) -> Signal {
            // Check which slot to use following Rust logic
            auto useSlot1 = If(newerSlot == UInt::constant(1, 1, builder, loc),
                [&](mlir::OpBuilder &innerBuilder) -> Signal {
                  // Use slot 0 if newer_slot == 1
                  slot0TxdReg->callMethod("write", {UInt::constant(1, 1, innerBuilder, loc).getValue()}, innerBuilder);
                  slot0TagReg->callMethod("write", {tagAddr.getValue()}, innerBuilder);
                  newerSlotReg->callMethod("write", {UInt::constant(0, 1, innerBuilder, loc).getValue()}, innerBuilder);
                  return UInt::constant(0, 1, innerBuilder, loc);
                },
                [&](mlir::OpBuilder &innerBuilder) -> Signal {
                  // Use slot 1 if newer_slot == 0
                  slot1TxdReg->callMethod("write", {UInt::constant(1, 1, innerBuilder, loc).getValue()}, innerBuilder);
                  slot1TagReg->callMethod("write", {tagAddr.getValue()}, innerBuilder);
                  newerSlotReg->callMethod("write", {UInt::constant(1, 1, innerBuilder, loc).getValue()}, innerBuilder);
                  return UInt::constant(0, 1, innerBuilder, loc);
                },
                builder, loc);
            return useSlot1;
          },
          [&](mlir::OpBuilder &builder) -> Signal {
            return UInt::constant(0, 1, builder, loc);
          },
          bodyBuilder, loc);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    respFromBus->finalize();

    // Method: resp_to_user - provides user memory responses (with ordering)
    // This matches the Rust method exactly - returns user response bundle to output interface
    llvm::SmallVector<std::pair<std::string, mlir::Type>, 0> respToUserArgs;
    llvm::SmallVector<mlir::Type, 1> respToUserReturns = {userRespBundleType};
    auto *respToUser = translatorModule->addMethod("resp_to_user", respToUserArgs, respToUserReturns);

    respToUser->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      // Check if any slot can collect (following Rust logic)
      auto slot0CollectValues = slot0CanCollectWire->callValue("read", guardBuilder);
      auto slot1CollectValues = slot1CanCollectWire->callValue("read", guardBuilder);
      Signal slot0CanCollect = Signal(slot0CollectValues[0], &guardBuilder, loc);
      Signal slot1CanCollect = Signal(slot1CollectValues[0], &guardBuilder, loc);

      auto hasResponse = slot0CanCollect | slot1CanCollect;
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, hasResponse.getValue());
    });

    respToUser->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      // Following Rust logic exactly: if_! (slot_0_can_collect.read() { ... } else { ... })
      auto slot0CollectValues = slot0CanCollectWire->callValue("read", bodyBuilder);
      Signal slot0CanCollect = Signal(slot0CollectValues[0], &bodyBuilder, loc);

      // Rust pattern: let retval = if_! (slot_0_can_collect.read() { ... } else { ... })
      auto retval = If(slot0CanCollect,
          [&](mlir::OpBuilder &builder) -> Signal {
            // Clear slot 0 flags BEFORE returning (matching Rust exactly)
            slot0TxdReg->callMethod("write", {UInt::constant(0, 1, builder, loc).getValue()}, builder);
            slot0RxdReg->callMethod("write", {UInt::constant(0, 1, builder, loc).getValue()}, builder);

            // Read slot 0 data for immediate use if needed
            auto data0Values = slot0DataReg->callValue("read", bodyBuilder);
            auto tag0Values = slot0TagReg->callValue("read", bodyBuilder);
            Signal data0 = Signal(data0Values[0], &bodyBuilder, loc);
            Signal tag0 = Signal(tag0Values[0], &bodyBuilder, loc);

            // Create UserMemoryResp bundle from unpacked fields using BundleCreateOp
            // Order must match the bundle type definition: data, tag
            llvm::SmallVector<mlir::Value> respBundleFields = {
              data0.getValue(), tag0.getValue()
            };

            auto respBundleValue = builder.create<BundleCreateOp>(loc, userRespBundleType, respBundleFields);
            return Signal(respBundleValue.getResult(), &builder, loc);
          },
          [&](mlir::OpBuilder &builder) -> Signal {
            // Else branch: clear slot 1 flags and check if slot 1 can collect
            slot1TxdReg->callMethod("write", {UInt::constant(0, 1, builder, loc).getValue()}, builder);
            slot1RxdReg->callMethod("write", {UInt::constant(0, 1, builder, loc).getValue()}, builder);

            // Read slot 1 data
            auto data1Values = slot1DataReg->callValue("read", builder);
            auto tag1Values = slot1TagReg->callValue("read", builder);
            Signal data1 = Signal(data1Values[0], &builder, loc);
            Signal tag1 = Signal(tag1Values[0], &builder, loc);

            // Create UserMemoryResp bundle from unpacked fields using BundleCreateOp
            // Order must match the bundle type definition: data, tag
            llvm::SmallVector<mlir::Value> respBundleFields = {
              data1.getValue(), tag1.getValue()
            };

            auto respBundleValue = builder.create<BundleCreateOp>(loc, userRespBundleType, respBundleFields);
            return Signal(respBundleValue.getResult(), &builder, loc);
          },
          bodyBuilder, loc);

      // Return the retval (matching Rust: ret!(var!(retval)))
      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, retval.getValue());
    });

    respToUser->finalize();

    // Add scheduling constraints (matching Rust schedule! calls)
    // Rust: schedule!(resp_from_bus, commit_cmd) means resp_from_bus must fire before commit_cmd
    // Rust: schedule!(resp_from_bus, resp_to_user) means resp_from_bus must fire before resp_to_user
    // Note: In CMT2 C++, this would be handled by the scheduling system, but we document it here
    translatorModule->setPrecedence({{"resp_from_bus", "commit_cmd"},{"resp_from_bus", "resp_to_user"}});
    // Schedule relationships:
    // 1. resp_from_bus → commit_cmd (resp_from_bus must execute before commit_cmd)
    // 2. resp_from_bus → resp_to_user (resp_from_bus must execute before resp_to_user)

    return translatorModule;
  }

  /// Generate RoCC Adapter module that bridges RoCC interface with accelerator execution units
  Module *generateRoCCAdapter(Circuit &circuit, const llvm::SmallVector<uint32_t, 4> &opcodes) {
    auto *roccAdapterModule = circuit.addModule("RoCCAdapter");
    auto &builder = roccAdapterModule->getBuilder();
    auto loc = roccAdapterModule->getLoc();
    auto *context = builder.getContext();

    // Add clock and reset arguments
    Clock clk = roccAdapterModule->addClockArgument("clk");
    Reset rst = roccAdapterModule->addResetArgument("rst");

    // Create bundle types following Rust patterns
    // RoccCmd: {funct: u7, rs1: u5, rs2: u5, rd: u5, xs1: u1, xs2: u1, xd: u1, opcode: u7, rs1data: u32, rs2data: u32}
    auto roccCmdBundleType = BundleType::get(context, {
      BundleType::BundleElement{builder.getStringAttr("funct"), false, UIntType::get(context, 7)},
      BundleType::BundleElement{builder.getStringAttr("rs1"), false, UIntType::get(context, 5)},
      BundleType::BundleElement{builder.getStringAttr("rs2"), false, UIntType::get(context, 5)},
      BundleType::BundleElement{builder.getStringAttr("rd"), false, UIntType::get(context, 5)},
      BundleType::BundleElement{builder.getStringAttr("xs1"), false, UIntType::get(context, 1)},
      BundleType::BundleElement{builder.getStringAttr("xs2"), false, UIntType::get(context, 1)},
      BundleType::BundleElement{builder.getStringAttr("xd"), false, UIntType::get(context, 1)},
      BundleType::BundleElement{builder.getStringAttr("opcode"), false, UIntType::get(context, 7)},
      BundleType::BundleElement{builder.getStringAttr("rs1data"), false, UIntType::get(context, 32)},
      BundleType::BundleElement{builder.getStringAttr("rs2data"), false, UIntType::get(context, 32)}
    });

    auto roccRespBundleType = BundleType::get(builder.getContext(), {
      BundleType::BundleElement{builder.getStringAttr("rd"), false, UIntType::get(context, 5)},
      BundleType::BundleElement{builder.getStringAttr("rddata"), false, UIntType::get(context, 32)}
    });

    // Create external FIFO modules
    auto savedIP = builder.saveInsertionPoint();

    // Calculate total bits for RoCC command bundle: 7+5+5+5+1+1+1+7+32+32 = 96 bits
    auto roccCmdFifoMod = STLLibrary::createFIFO1PushModule(96, circuit);

    // Calculate total bits for RoCC response bundle: 5+32 = 37 bits (padded to appropriate width)
    auto roccRespFifoMod = STLLibrary::createFIFO1PushModule(37, circuit);

    builder.restoreInsertionPoint(savedIP);

    // Create FIFO instances for each opcode
    llvm::SmallVector<Instance*, 4> roccCmdFifos;
    for (uint32_t opcode : opcodes) {
      auto *fifo = roccAdapterModule->addInstance(
          "rocc_cmd_fifo_" + std::to_string(opcode),
          roccCmdFifoMod,
          {clk.getValue(), rst.getValue()}
      );
      roccCmdFifos.push_back(fifo);
    }

    // Create response FIFO instance
    auto *roccRespFifo = roccAdapterModule->addInstance("rocc_resp_fifo", roccRespFifoMod,
                                                        {clk.getValue(), rst.getValue()});

    // Method: cmd_from_bus - receives RoCC commands from the bus and routes to appropriate opcode queues
    llvm::SmallVector<std::pair<std::string, mlir::Type>, 1> cmdFromBusArgs;
    cmdFromBusArgs.push_back({"rocc_cmd_bus", roccCmdBundleType});
    auto *cmdFromBus = roccAdapterModule->addMethod("cmd_from_bus", cmdFromBusArgs, {});

    cmdFromBus->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument>) {
      auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
    });

    cmdFromBus->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      Bundle roccCmd(args[0], &bodyBuilder, loc);

      // Extract all fields from RoCC command bundle
      Signal funct = roccCmd["funct"];
      Signal rs1 = roccCmd["rs1"];
      Signal rs2 = roccCmd["rs2"];
      Signal rd = roccCmd["rd"];
      Signal xs1 = roccCmd["xs1"];
      Signal xs2 = roccCmd["xs2"];
      Signal xd = roccCmd["xd"];
      Signal opcode = roccCmd["opcode"];
      Signal rs1data = roccCmd["rs1data"];
      Signal rs2data = roccCmd["rs2data"];

      // Pack the bundle into a 96-bit concatenated value for FIFO
      // Layout: rs2data(32) + rs1data(32) + opcode(7) + xd(1) + xs2(1) + xs1(1) + rd(5) + rs2(5) + rs1(5) + funct(7) = 96 bits
      Signal packedCmd = rs2data.cat(rs1data).cat(opcode).cat(xd).cat(xs2).cat(xs1)
                                .cat(rd).cat(rs2).cat(rs1).cat(funct);

      // Route command to appropriate queue based on opcode (following Rust pattern)
      for (size_t i = 0; i < opcodes.size(); ++i) {
        uint32_t targetOpcode = opcodes[i];
        auto *fifo = roccCmdFifos[i];

        // Check if this command's opcode matches the target
        auto opcodeMatch = opcode == UInt::constant(targetOpcode, 7, bodyBuilder, loc);

        // Conditional enqueue (matching Rust if_! pattern)
        If(opcodeMatch,
            [&](mlir::OpBuilder &innerBuilder) -> Signal {
              // Enqueue the packed 96-bit command to the appropriate FIFO
              fifo->callMethod("enq", {packedCmd.getValue()}, innerBuilder);
              return UInt::constant(0, 1, innerBuilder, loc);
            },
            [&](mlir::OpBuilder &innerBuilder) -> Signal {
              return UInt::constant(0, 1, innerBuilder, loc);
            },
            bodyBuilder, loc);
      }

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    cmdFromBus->finalize();

    // Method: resp_from_user - receives responses from user execution units and enqueues for return
    llvm::SmallVector<std::pair<std::string, mlir::Type>, 1> respFromUserArgs;
    respFromUserArgs.push_back({"rocc_resp_user", roccRespBundleType});
    auto *respFromUser = roccAdapterModule->addMethod("resp_from_user", respFromUserArgs, {});

    respFromUser->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument>) {
      auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
    });

    respFromUser->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
      Bundle roccResp(args[0], &bodyBuilder, loc);

      // Extract fields from RoCC response bundle
      Signal rd = roccResp["rd"];
      Signal rddata = roccResp["rddata"];

      // Pack the bundle into a 37-bit concatenated value for FIFO
      // Layout: rddata(32) + rd(5) = 37 bits
      Signal packedResp = rddata.cat(rd);

      // Enqueue the packed response to the response FIFO
      roccRespFifo->callMethod("enq", {packedResp.getValue()}, bodyBuilder);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    respFromUser->finalize();

    // Rule: commit - processes response queue and sends to RoCC master
    auto *commitRule = roccAdapterModule->addRule("commit");

    commitRule->guard([&](mlir::OpBuilder &guardBuilder) {
      auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
    });

    commitRule->body([&](mlir::OpBuilder &bodyBuilder) {
      // Dequeue response from FIFO (37-bit concatenated value)
      auto respValues = roccRespFifo->callValue("deq", bodyBuilder);
      Signal packedResp(respValues[0], &bodyBuilder, loc);

      // Unpack the 37-bit concatenated value back into bundle fields
      // Layout: rddata(32) + rd(5) = 37 bits
      Signal rddata = packedResp.bits(36, 5);  // bits 5-36: rddata (32 bits)
      Signal rd = packedResp.bits(4, 0);       // bits 0-4: rd (5 bits)

      // Create RoCC response bundle from unpacked fields using BundleCreateOp
      // Order must match the bundle type definition: rd, rddata
      llvm::SmallVector<mlir::Value> respBundleFields = {
        rd.getValue(), rddata.getValue()
      };

      auto respBundleValue = bodyBuilder.create<BundleCreateOp>(loc, roccRespBundleType, respBundleFields);

      // TODO: In a complete implementation, this would call the RoCC master's resp_to_bus method
      // For now, we just consume the response from the queue
      // rocc_master.resp_to_bus(respBundleValue.getResult());
      (void)respBundleValue; // Suppress unused variable warning

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    commitRule->finalize();

    // Add cmd_to_user methods for each opcode (following Rust pattern exactly)
    for (size_t i = 0; i < opcodes.size(); ++i) {
      uint32_t opcode = opcodes[i];
      std::string methodName = "cmd_to_user_" + std::to_string(opcode);

      llvm::SmallVector<std::pair<std::string, mlir::Type>, 0> cmdToUserArgs;
      llvm::SmallVector<mlir::Type, 1> cmdToUserReturns = {roccCmdBundleType};
      auto *cmdToUser = roccAdapterModule->addMethod(methodName, cmdToUserArgs, cmdToUserReturns);

      cmdToUser->guard([&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
        auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
        guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
      });

      cmdToUser->body([&](mlir::OpBuilder &bodyBuilder, llvm::ArrayRef<mlir::BlockArgument> args) {
        // Dequeue command from the appropriate FIFO (96-bit concatenated value)
        auto cmdValues = roccCmdFifos[i]->callValue("deq", bodyBuilder);
        Signal packedCmd(cmdValues[0], &bodyBuilder, loc);

        // Unpack the 96-bit concatenated value back into bundle fields
        // Layout: rs2data(32) + rs1data(32) + opcode(7) + xd(1) + xs2(1) + xs1(1) + rd(5) + rs2(5) + rs1(5) + funct(7) = 96 bits
        Signal rs2data = packedCmd.bits(95, 64);   // bits 64-95: rs2data (32 bits)
        Signal rs1data = packedCmd.bits(63, 32);   // bits 32-63: rs1data (32 bits)
        Signal opcode = packedCmd.bits(31, 25);    // bits 25-31: opcode (7 bits)
        Signal xd = packedCmd.bits(24, 24);        // bit 24: xd (1 bit)
        Signal xs2 = packedCmd.bits(23, 23);       // bit 23: xs2 (1 bit)
        Signal xs1 = packedCmd.bits(22, 22);       // bit 22: xs1 (1 bit)
        Signal rd = packedCmd.bits(21, 17);        // bits 17-21: rd (5 bits)
        Signal rs2 = packedCmd.bits(16, 12);       // bits 12-16: rs2 (5 bits)
        Signal rs1 = packedCmd.bits(11, 7);        // bits 7-11: rs1 (5 bits)
        Signal funct = packedCmd.bits(6, 0);        // bits 0-6: funct (7 bits)

        // Create RoCC command bundle from unpacked fields using BundleCreateOp
        // Order must match the bundle type definition: funct, rs1, rs2, rd, xs1, xs2, xd, opcode, rs1data, rs2data
        llvm::SmallVector<mlir::Value> bundleFields = {
          funct.getValue(), rs1.getValue(), rs2.getValue(), rd.getValue(),
          xs1.getValue(), xs2.getValue(), xd.getValue(), opcode.getValue(),
          rs1data.getValue(), rs2data.getValue()
        };

        auto bundleValue = bodyBuilder.create<BundleCreateOp>(loc, roccCmdBundleType, bundleFields);

        // Return the unpacked command bundle
        bodyBuilder.create<circt::cmt2::ReturnOp>(loc, bundleValue.getResult());
      });

      cmdToUser->finalize();
    }

    return roccAdapterModule;
  }

  /// Generate rules for a specific TOR function - proper implementation from rulegenpass.cpp
  void generateRulesForFunction(Module *mainModule, tor::FuncOp funcOp,
                               Instance *poolInstance, Instance *roccInstance, Instance *hellaMemInstance,
                               InterfaceDecl *dmaItfc, 
                               Circuit &circuit, Clock mainClk, Reset mainRst, uint32_t opcode) {
    
    auto &builder = mainModule->getBuilder();
    MLIRContext *ctx = builder.getContext();

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
        if (isa<tor::AddIOp, tor::SubIOp, tor::MulIOp>(op))
          continue;
        if (isa<aps::MemLoad, aps::CpuRfRead, aps::CpuRfWrite>(op))
          continue;
        op->emitError("unsupported operation for rule generation");
        return;
      }
    }

    // Analyze cross-slot value uses - simplified with single data structure
    llvm::DenseMap<mlir::Value, CrossSlotFIFO*> crossSlotFIFOs;
    llvm::DenseMap<std::pair<int64_t, int64_t>, unsigned> fifoCounts;
    SmallVector<std::unique_ptr<CrossSlotFIFO>, 8> fifoStorage;
    
    auto getSlotForOp = [&](Operation *op) -> std::optional<int64_t> {
      if (auto attr = op->getAttrOfType<IntegerAttr>("starttime"))
        return attr.getInt();
      return {};
    };

    // Build cross-slot FIFO mapping - single pass analysis
    for (int64_t slot : slotOrder) {
      for (Operation *op : slotMap[slot].ops) {
        if (isa<arith::ConstantOp>(op))
          continue;

        for (mlir::Value res : op->getResults()) {
          if (!isa<mlir::IntegerType>(res.getType()))
            continue;

          // Check if this value has cross-slot uses
          bool hasCrossSlotUses = false;
          for (OpOperand &use : res.getUses()) {
            Operation *user = use.getOwner();
            auto maybeConsumerSlot = getSlotForOp(user);
            if (!maybeConsumerSlot)
              continue;
            int64_t consumerSlot = *maybeConsumerSlot;
            if (consumerSlot > slot) {
              hasCrossSlotUses = true;
              break;
            }
          }

          if (!hasCrossSlotUses)
            continue;

          auto firType = toFirrtlType(res.getType(), ctx);
          if (!firType) {
            op->emitError("type is unsupported for rule lowering");
            return;
          }

          // Create single FIFO for this producer value
          auto fifo = std::make_unique<CrossSlotFIFO>();
          fifo->producerValue = res;
          fifo->producerSlot = slot;
          fifo->firType = firType;

          // Collect all consumers for this producer
          for (OpOperand &use : res.getUses()) {
            Operation *user = use.getOwner();
            auto maybeConsumerSlot = getSlotForOp(user);
            if (!maybeConsumerSlot)
              continue;
            int64_t consumerSlot = *maybeConsumerSlot;
            if (consumerSlot > slot) {
              fifo->consumers.push_back({user, use.getOperandNumber()});
              if (fifo->consumerSlot == 0)  // Set primary consumer slot
                fifo->consumerSlot = consumerSlot;
            }
          }

          // Generate FIFO name based on producer-consumer slot pair
          auto key = std::make_pair(slot, fifo->consumerSlot);
          unsigned count = fifoCounts[key]++;
          fifo->instanceName = std::to_string(opcode) + "_fifo_s" + std::to_string(slot) + "_s" +
                               std::to_string(fifo->consumerSlot);
          if (count > 0)
            fifo->instanceName += "_" + std::to_string(count);

          crossSlotFIFOs[res] = fifo.get();
          fifoStorage.push_back(std::move(fifo));
        }
      }
    }

    auto savedIP = builder.saveInsertionPoint();
    // Instantiate FIFO modules for cross-slot communication
    for (auto &fifoPtr : fifoStorage) {
      auto *fifo = fifoPtr.get();
      int64_t width = cast<circt::firrtl::UIntType>(fifo->firType).getWidthOrSentinel();
      if (width < 0)
        width = 1;

      // Create FIFO module with proper clock and reset
      auto *fifoMod = STLLibrary::createFIFO1PushModule(width, circuit);
      builder.restoreInsertionPoint(savedIP);
      fifo->fifoInstance = mainModule->addInstance(fifo->instanceName, fifoMod,
                                                   {mainClk.getValue(), mainRst.getValue()});
    }

    // Create token FIFOs for stage synchronization - one token per stage (except last)
    llvm::DenseMap<int64_t, Instance*> stageTokenFifos;
    for (size_t i = 0; i < slotOrder.size() - 1; ++i) {
      int64_t currentSlot = slotOrder[i];
      // Create 1-bit FIFO for token passing to next stage
      auto *tokenFifoMod = STLLibrary::createFIFO1PushModule(1, circuit);
      builder.restoreInsertionPoint(savedIP);
      std::string tokenFifoName = std::to_string(opcode) + "_token_fifo_s" + std::to_string(currentSlot);
      auto *tokenFifo = mainModule->addInstance(tokenFifoName, tokenFifoMod,
                                                {mainClk.getValue(), mainRst.getValue()});
      stageTokenFifos[currentSlot] = tokenFifo;
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
        if (sourceOp) sourceOp->emitError("expected FIRRTL UInt type");
        return failure();
      }

      int64_t currentWidth = type.getWidthOrSentinel();
      if (currentWidth < 0) {
        if (sourceOp) sourceOp->emitError("requires statically known bitwidth");
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

      // Check if this is a cross-slot FIFO read (only if operandIndex is valid)
      if (operandIndex != static_cast<unsigned>(-1)) {
        auto it = crossSlotFIFOs.find(v);
        if (it != crossSlotFIFOs.end()) {
          CrossSlotFIFO *fifo = it->second;
          // Check if this operation is a consumer of this FIFO
          for (auto [consumerOp, opIndex] : fifo->consumers) {
            if (consumerOp == currentOp && opIndex == operandIndex) {
              auto result = fifo->fifoInstance->callValue("deq", builder);
              assert(result.size() == 1);
              localMap[v] = result[0];
              return result[0];
            }
          }
        }
      }

      currentOp->emitError("value is not available in this rule");
      return failure();
    };

    // Generate rules for each time slot.
    llvm::DenseMap<mlir::Value, mlir::Value> localMap;
    llvm::SmallVector<std::pair<std::string, std::string>, 4> precedencePairs;

    // Cache for RoCC command bundle (pre-read in slot 0)
    mlir::Value cachedRoCCCmdBundle;

    auto savedIPforRegRd = builder.saveInsertionPoint();
    auto *regRdMod = STLLibrary::createRegModule(5, 0, circuit);
    auto *regRdInstance = mainModule->addInstance("reg_rd_" + std::to_string(opcode), regRdMod, {mainClk.getValue(), mainRst.getValue()});
    builder.restoreInsertionPoint(savedIPforRegRd);

    // Unified arithmetic operation helper to eliminate redundancy
    auto performArithmeticOp = [&](mlir::OpBuilder &b, Location loc, mlir::Value lhs, mlir::Value rhs,
                                    mlir::Value result, StringRef opName) -> LogicalResult {
      // Determine result width based on operation type
      auto lhsWidth = cast<circt::firrtl::UIntType>(lhs.getType()).getWidthOrSentinel();
      auto rhsWidth = cast<circt::firrtl::UIntType>(rhs.getType()).getWidthOrSentinel();

      int64_t resultWidth;
      if (opName == "mul") {
        resultWidth = lhsWidth + rhsWidth;  // Multiplication doubles width
      } else {
        resultWidth = std::max(lhsWidth, rhsWidth);  // Add/sub use max width
      }

      // Ensure both operands have the same width
      auto lhsExtended = ensureUIntWidth(b, loc, lhs, resultWidth, nullptr);
      auto rhsExtended = ensureUIntWidth(b, loc, rhs, resultWidth, nullptr);
      if (failed(lhsExtended) || failed(rhsExtended))
        return failure();

      // Perform the arithmetic operation using Signal abstraction
      Signal lhsSignal(*lhsExtended, &b, loc);
      Signal rhsSignal(*rhsExtended, &b, loc);
      mlir::Value resultValue;

      if (opName == "add") {
        resultValue = (lhsSignal + rhsSignal).getValue();
      } else if (opName == "sub") {
        resultValue = (lhsSignal - rhsSignal).getValue();
      } else if (opName == "mul") {
        resultValue = (lhsSignal * rhsSignal).getValue();
      } else {
        return failure();
      }

      localMap[result] = resultValue;
      return success();
    };

    // auto trueConst = builder.create<ConstantOp>(mainModule->getLoc(), u64Type, llvm::APInt(64, 1927));
    // auto builder2 = mainModule->getBuilder();
    // builder2.create<ConstantOp>(mainModule->getLoc(), u64Type, llvm::APInt(64, 1949));
    for (int64_t slot : slotOrder) {
      auto *rule = mainModule->addRule(std::to_string(opcode) + "_rule_s" + std::to_string(slot));

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

        // Pre-read RoCC command bundle in slot 0
        if (slot == 0) {
          // Call cmd_to_user once to get the RoCC command bundle
          std::string cmdMethod = "cmd_to_user_" + std::to_string(opcode);
          auto cmdResult = roccInstance->callMethod(cmdMethod, {}, b)[0];
          cachedRoCCCmdBundle = cmdResult;
          auto instruction = Bundle(cachedRoCCCmdBundle, &b, loc);
          regRdInstance->callMethod("write", {instruction["rd"].getValue()}, b);
        }

        // Read token from previous stage at the start (except for first stage)
        if (!slotOrder.empty() && slot != slotOrder[0]) {
          auto it = std::find(slotOrder.begin(), slotOrder.end(), slot);
          if (it != slotOrder.begin()) {
            int64_t prevSlot = *(it - 1);
            auto tokenFifoIt = stageTokenFifos.find(prevSlot);
            if (tokenFifoIt != stageTokenFifos.end()) {
              auto *tokenFifo = tokenFifoIt->second;
              auto tokenValues = tokenFifo->callValue("deq", b);
              // Token value should be 1'b1, but we don't need to use it
              // Just reading from FIFO ensures synchronization
            }
          }
        }

        for (Operation *op : slotMap[slot].ops) {
          // Handle arithmetic operations using unified helper
          if (auto addOp = dyn_cast<tor::AddIOp>(op)) {
            auto lhs = getValueInRule(addOp.getLhs(), op, 0, b, localMap, loc);
            auto rhs = getValueInRule(addOp.getRhs(), op, 1, b, localMap, loc);
            if (failed(lhs) || failed(rhs))
              return;
            if (failed(performArithmeticOp(b, loc, *lhs, *rhs, addOp.getResult(), "add")))
              return;
          }
          else if (auto subOp = dyn_cast<tor::SubIOp>(op)) {
            auto lhs = getValueInRule(subOp.getLhs(), op, 0, b, localMap, loc);
            auto rhs = getValueInRule(subOp.getRhs(), op, 1, b, localMap, loc);
            if (failed(lhs) || failed(rhs))
              return;
            if (failed(performArithmeticOp(b, loc, *lhs, *rhs, subOp.getResult(), "sub")))
              return;
          }
          else if (auto mulOp = dyn_cast<tor::MulIOp>(op)) {
            auto lhs = getValueInRule(mulOp.getLhs(), op, 0, b, localMap, loc);
            auto rhs = getValueInRule(mulOp.getRhs(), op, 1, b, localMap, loc);
            if (failed(lhs) || failed(rhs))
              return;
            if (failed(performArithmeticOp(b, loc, *lhs, *rhs, mulOp.getResult(), "mul")))
              return;
          }
          else if (auto readRf = dyn_cast<aps::CpuRfRead>(op)) {
            // Handle CPU register file read operations
            // PANIC if not in first time slot
            if (slot != 0) {
              op->emitError("aps.readrf operations must appear only in the first time slot (slot 0), but found in slot ") << slot;
              llvm::report_fatal_error("readrf must be in first time slot");
            }
            // Get the function argument that this readrf is reading from
            Value regArg = readRf.getRs();

            // Map function arguments to rs1/rs2 based on their position in the function
            // Need to find which argument index this corresponds to
            int64_t argIndex = -1;
            for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
              if (funcOp.getArgument(i) == regArg) {
                argIndex = i;
                break;
              }
            }
            auto instruction = Bundle(cachedRoCCCmdBundle, &b, loc);
            Signal regValue = argIndex == 0 ? instruction["rs1data"] : instruction["rs2data"];
            // // Store the result with a note about which argument it came from
            // llvm::outs() << "DEBUG: readrf from arg" << argIndex << " using cached RoCC bundle\n";
            localMap[readRf.getResult()] = regValue.getValue();
          }
          else if (auto writeRf = dyn_cast<aps::CpuRfWrite>(op)) {
            auto rdvalue = getValueInRule(writeRf.getValue(), op, 1, b, localMap, loc);
            // auto rd = UInt::constant(1, 5, b, loc);
            auto rd = regRdInstance->callValue("read", b)[0];
            llvm::SmallVector<mlir::Value> bundleFields = {rd, *rdvalue};
            auto bundleType = BundleType::get(b.getContext(), {
              BundleType::BundleElement{b.getStringAttr("rd"), false, UIntType::get(b.getContext(), 5)},
              BundleType::BundleElement{b.getStringAttr("rddata"), false, UIntType::get(b.getContext(), 32)},
            });

            auto bundleValue = b.create<BundleCreateOp>(loc, bundleType, bundleFields);
            roccInstance->callMethod("resp_from_user", {bundleValue}, b);
            // writerf doesn't produce a result, just performs the write
          }
          else if (auto itfcLoadReq = dyn_cast<aps::ItfcLoadReq>(op)) {
            // Handle CPU interface load request operations
            // This sends a load request to memory and returns a request token
            auto context = b.getContext();

            auto userCmdBundleType = BundleType::get(context, {
              BundleType::BundleElement{b.getStringAttr("addr"), false, UIntType::get(context, 32)},
              BundleType::BundleElement{b.getStringAttr("cmd"), false, UIntType::get(context, 1)},
              BundleType::BundleElement{b.getStringAttr("size"), false, UIntType::get(context, 2)},
              BundleType::BundleElement{b.getStringAttr("data"), false, UIntType::get(context, 32)},
              BundleType::BundleElement{b.getStringAttr("mask"), false, UIntType::get(context, 4)},
              BundleType::BundleElement{b.getStringAttr("tag"), false, UIntType::get(context, 8)}
            });

            // Get address from indices (similar to MemLoad)
            if (itfcLoadReq.getIndices().empty()) {
              op->emitError("Interface load request must have at least one index");
            }

            auto addr = getValueInRule(itfcLoadReq.getIndices()[0], op, 0, b, localMap, loc);
            if (failed(addr)) {
              op->emitError("Failed to get address for interface load request");
            }
            auto addrSignal = Signal(*addr, &b, loc);
            auto readCmd = UInt::constant(0, 1, b, loc);
            auto data = UInt::constant(0, 32, b, loc);
            auto size = UInt::constant(2, 2, b, loc);
            auto mask = UInt::constant(0, 4, b, loc);
            auto tagConst = UInt::constant(0x3f, 8, b, loc);
            auto tag = (addrSignal ^ tagConst).bits(7, 0);

            llvm::SmallVector<mlir::Value> bundleFields = {*addr, readCmd.getValue(), size.getValue(), data.getValue(), mask.getValue(), tag.getValue()};
            auto bundleValue = b.create<BundleCreateOp>(loc, userCmdBundleType, bundleFields);
            hellaMemInstance->callMethod("cmd_from_user", {bundleValue}, b);

            localMap[itfcLoadReq.getResult()] = UInt::constant(1, 1, b, loc).getValue();
          }
          else if (auto itfcLoadCollect = dyn_cast<aps::ItfcLoadCollect>(op)) {
            auto resp = hellaMemInstance->callMethod("resp_to_user", {}, b)[0];
            auto respBundle = Bundle(resp, &b, loc);

            auto loadResult = respBundle["data"];

            localMap[itfcLoadCollect.getResult()] = loadResult.getValue();
          }
          else if (auto itfcStoreReq = dyn_cast<aps::ItfcStoreReq>(op)) {
            // Handle CPU interface store request operations
            // This sends a store request to memory and returns a request token

            // Get address from indices (value is first operand, address is from indices)
            auto context = b.getContext();
            if (itfcStoreReq.getIndices().empty()) {
              op->emitError("Interface store request must have at least one index");
            }
            auto value = getValueInRule(itfcStoreReq.getValue(), op, 0, b, localMap, loc);
            auto addr = getValueInRule(itfcStoreReq.getIndices()[0], op, 1, b, localMap, loc);
            if (failed(value) || failed(addr)) {
              op->emitError("Failed to get value or address for interface store request");
            }

            auto userCmdBundleType = BundleType::get(context, {
              BundleType::BundleElement{b.getStringAttr("addr"), false, UIntType::get(context, 32)},
              BundleType::BundleElement{b.getStringAttr("cmd"), false, UIntType::get(context, 1)},
              BundleType::BundleElement{b.getStringAttr("size"), false, UIntType::get(context, 2)},
              BundleType::BundleElement{b.getStringAttr("data"), false, UIntType::get(context, 32)},
              BundleType::BundleElement{b.getStringAttr("mask"), false, UIntType::get(context, 4)},
              BundleType::BundleElement{b.getStringAttr("tag"), false, UIntType::get(context, 8)}
            });

            if (failed(addr)) {
              op->emitError("Failed to get address for interface load request");
            }
            auto addrSignal = Signal(*addr, &b, loc);
            auto WriteCmd = UInt::constant(1, 1, b, loc);
            auto size = UInt::constant(2, 2, b, loc);
            auto mask = UInt::constant(0, 4, b, loc);
            auto tagConst = UInt::constant(0x3f, 8, b, loc);
            auto tag = (addrSignal ^ tagConst).bits(7, 0);

            llvm::SmallVector<mlir::Value> bundleFields = {*addr, WriteCmd.getValue(), size.getValue(), *value, mask.getValue(), tag.getValue()};
            auto bundleValue = b.create<BundleCreateOp>(loc, userCmdBundleType, bundleFields);
            hellaMemInstance->callMethod("cmd_from_user", {bundleValue}, b);

            localMap[itfcStoreReq.getResult()] = UInt::constant(1, 1, b, loc).getValue();
          }
          else if (auto itfcStoreCollect = dyn_cast<aps::ItfcStoreCollect>(op)) {
            // Do nothing here, don't reply...
          }
          else if (auto memLoad = dyn_cast<aps::MemLoad>(op)) {
            // Handle memory load operations
            // PANIC if no indices provided
            if (memLoad.getIndices().empty()) {
              op->emitError("Memory load operation must have at least one index");
              llvm::report_fatal_error("Memory load requires address indices");
            }

            auto addr = getValueInRule(memLoad.getIndices()[0], op, 0, b, localMap, loc);
            if (failed(addr)) {
              op->emitError("Failed to get address for memory load");
              llvm::report_fatal_error("Memory load address resolution failed");
            }

            // Get the memory reference and check if it comes from memref.get_global
            Value memRef = memLoad.getMemref();
            Operation *defOp = memRef.getDefiningOp();

            std::string memoryBankRule;
            if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(defOp)) {
              // Extract the global symbol name and build the rule name
              StringRef globalName = getGlobalOp.getName();
              // Convert @mem_a_0 -> mem_a_0_read
              memoryBankRule = (globalName.drop_front() + "_read").str(); // Remove @ prefix and add _read
              llvm::outs() << "DEBUG: Memory load from global " << globalName << " using rule " << memoryBankRule << "\n";
            }

            // Call the appropriate scratchpad pool bank read method
            auto callResult = b.create<circt::cmt2::CallOp>(
                loc, circt::firrtl::UIntType::get(b.getContext(), 64),
                mlir::ValueRange{*addr},
                mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
                mlir::SymbolRefAttr::get(b.getContext(), memoryBankRule),
                mlir::ArrayAttr(), mlir::ArrayAttr());

            localMap[memLoad.getResult()] = callResult.getResult(0);
          } else if (auto memBurstLoadReq = dyn_cast<aps::ItfcBurstLoadReq>(op)) {
            Value cpuAddr = memBurstLoadReq.getCpuAddr();
            Value memRef = memBurstLoadReq.getMemrefs()[0];
            Value start = memBurstLoadReq.getStart();
            Value numOfElements = memBurstLoadReq.getLength();

            // Get the global memory reference name
            auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(memRef.getDefiningOp());
            auto globalName = getGlobalOp.getName().str();

            // Find the corresponding memory entry using the pre-built map
            MemoryEntryInfo* targetMemEntry = nullptr;
            if (!globalName.empty()) {
              auto it = memEntryMap.find(globalName);
              if (it != memEntryMap.end()) {
                targetMemEntry = &it->second;
              }
            }

            // Get element type and size from the memory entry info or global memref
            int elementSizeBytes = 1; // Default to 1 byte
            if (targetMemEntry) {
              // Use the data width from the memory entry info
              elementSizeBytes = (targetMemEntry->dataWidth + 7) / 8; // Convert bits to bytes, rounding up
            } else if (!globalName.empty()) {
              op->emitError("Failed to find target memory entry!");
            }

            // Calculate localAddr: baseAddress + (start * numOfElements * elementSizeBytes)
            uint32_t baseAddress = targetMemEntry->baseAddress;

            // Calculate offset: start * numOfElements * elementSizeBytes
            // First multiply start * numOfElements
            auto baseAddrConst = UInt::constant(baseAddress, 32, b, loc);
            auto startSig = Signal(start, &b, loc);
            auto elementSizeBytesConst = UInt::constant(32, elementSizeBytes, b, loc);
            Signal localAddr = baseAddrConst + startSig * elementSizeBytesConst;

            // Calculate total burst length: elementSizeBytes * numOfElements, rounded up to nearest power of 2
            auto numElementsOp = numOfElements.getDefiningOp<arith::ConstantOp>();
            auto numElementsAttr = numElementsOp.getValue();
            auto numElements = dyn_cast<IntegerAttr>(numElementsAttr).getValue().getZExtValue();

            uint64_t totalBurstLength = (uint64_t)elementSizeBytes * numElements;
            uint32_t roundedTotalBurstLength = roundUpToPowerOf2((uint32_t)totalBurstLength);
            auto realCpuLength = UInt::constant(32, roundedTotalBurstLength, b, loc);

            dmaItfc->callMethod("cpu_to_isax", {
              cpuAddr, 
              localAddr.bits(31, 0).getValue(), 
              realCpuLength.bits(3, 0).getValue()
            }, b);

          } else if (auto memBurstLoadCollect = dyn_cast<aps::ItfcBurstLoadCollect>(op)) {
            // no action needed
            dmaItfc->callMethod("poll_for_idle", {}, b);
          } else if (auto memBurstStoreReq = dyn_cast<aps::ItfcBurstStoreReq>(op)) {

            Value cpuAddr = memBurstLoadReq.getCpuAddr();
            Value memRef = memBurstLoadReq.getMemrefs()[0];
            Value start = memBurstLoadReq.getStart();
            Value numOfElements = memBurstLoadReq.getLength();

            // Get the global memory reference name
            auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(memRef.getDefiningOp());
            auto globalName = getGlobalOp.getName().str();

            // Find the corresponding memory entry using the pre-built map
            MemoryEntryInfo* targetMemEntry = nullptr;
            if (!globalName.empty()) {
              auto it = memEntryMap.find(globalName);
              if (it != memEntryMap.end()) {
                targetMemEntry = &it->second;
              }
            }

            // Get element type and size from the memory entry info or global memref
            int elementSizeBytes = 1; // Default to 1 byte
            if (targetMemEntry) {
              // Use the data width from the memory entry info
              elementSizeBytes = (targetMemEntry->dataWidth + 7) / 8; // Convert bits to bytes, rounding up
            } else if (!globalName.empty()) {
              op->emitError("Failed to find target memory entry!");
            }

            // Calculate localAddr: baseAddress + (start * numOfElements * elementSizeBytes)
            uint32_t baseAddress = targetMemEntry->baseAddress;

            // Calculate offset: start * numOfElements * elementSizeBytes
            // First multiply start * numOfElements
            auto baseAddrConst = UInt::constant(baseAddress, 32, b, loc);
            auto startSig = Signal(start, &b, loc);
            auto elementSizeBytesConst = UInt::constant(32, elementSizeBytes, b, loc);
            Signal localAddr = baseAddrConst + startSig * elementSizeBytesConst;

            // Calculate total burst length: elementSizeBytes * numOfElements, rounded up to nearest power of 2
            auto numElementsOp = numOfElements.getDefiningOp<arith::ConstantOp>();
            auto numElementsAttr = numElementsOp.getValue();
            auto numElements = dyn_cast<IntegerAttr>(numElementsAttr).getValue().getZExtValue();

            uint64_t totalBurstLength = (uint64_t)elementSizeBytes * numElements;
            uint32_t roundedTotalBurstLength = roundUpToPowerOf2((uint32_t)totalBurstLength);
            auto realCpuLength = UInt::constant(32, roundedTotalBurstLength, b, loc);

            dmaItfc->callMethod("isax_to_cpu", {
              cpuAddr, 
              localAddr.bits(31, 0).getValue(), 
              realCpuLength.bits(3, 0).getValue()
            }, b);

          } else if (auto memBurstStoreCollect = dyn_cast<aps::ItfcBurstStoreCollect>(op)) {
            dmaItfc->callMethod("poll_for_idle", {}, b);
          }
          // Handle cross-slot FIFO writes for producer operations
          // Enqueue producer result values to appropriate FIFOs
          for (mlir::Value result : op->getResults()) {
            if (!isa<mlir::IntegerType>(result.getType()))
              continue;

            auto fifoIt = crossSlotFIFOs.find(result);
            if (fifoIt != crossSlotFIFOs.end()) {
              CrossSlotFIFO *fifo = fifoIt->second;
              // Enqueue the value to the FIFO in the producer's slot
              auto value = getValueInRule(result, op, /*operandIndex=*/-1, b, localMap, loc);
              if (failed(value))
                return;
              fifo->fifoInstance->callMethod("enq", {*value}, b);
            }
          }
        }

        // Write token to next stage at the end (except for last stage)
        if (!slotOrder.empty() && slot != slotOrder.back()) {
          auto tokenFifoIt = stageTokenFifos.find(slot);
          if (tokenFifoIt != stageTokenFifos.end()) {
            auto *tokenFifo = tokenFifoIt->second;
            auto tokenValue = UInt::constant(1, 1, b, loc);
            tokenFifo->callMethod("enq", {tokenValue.getValue()}, b);
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
          std::string from = std::to_string(opcode) + "_rule_s" + std::to_string(slot);
          std::string to = std::to_string(opcode) + "_rule_s" + std::to_string(prevSlot);
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
