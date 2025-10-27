#include "APS/APSToCMT2.h"

#define DEBUG_TYPE "aps-memory-pool-gen"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

/// Extract data width and address width from a memref type
bool APSToCMT2GenPass::extractMemoryParameters(memref::GlobalOp globalOp,
                                               int &dataWidth, int &addrWidth,
                                               int &depth) {
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

// ============================================================================
//
//                                BANK WRAPPER
//
// ============================================================================

/// Generate a bank wrapper module that encapsulates bank selection and data
/// alignment
Module *APSToCMT2GenPass::generateBankWrapperModule(
    const MemoryEntryInfo &entryInfo, Circuit &circuit, size_t bankIdx,
    ExternalModule *memMod, Clock clk, Reset rst) {
  std::string wrapperName =
      "BankWrapper_" + entryInfo.name + "_" + std::to_string(bankIdx);
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
      "mem_bank", memMod, {wrapperClk.getValue(), wrapperRst.getValue()});

  // Create wire_default modules for enable, data, and addr
  // Save insertion point before creating wire modules
  auto savedIPForWires = builder.saveInsertionPoint();

  auto wireEnableMod = STLLibrary::createWireDefaultModule(1, 0, circuit);
  auto wireDataMod = STLLibrary::createWireModule(entryInfo.dataWidth, circuit);
  auto wireAddrMod = STLLibrary::createWireModule(entryInfo.addrWidth, circuit);

  // Restore insertion point back to wrapper
  builder.restoreInsertionPoint(savedIPForWires);

  // Create wire_default instances in the wrapper
  auto *writeEnableWire =
      wrapper->addInstance("write_enable_wire", wireEnableMod, {});
  auto *writeDataWire =
      wrapper->addInstance("write_data_wire", wireDataMod, {});
  auto *writeAddrWire =
      wrapper->addInstance("write_addr_wire", wireAddrMod, {});

  // burst_read method: returns 64-bit aligned data if address is for this bank,
  // else 0
  auto *burstRead =
      wrapper->addMethod("burst_read", {{"addr", u64Type}}, {u64Type});

  burstRead->guard([&](mlir::OpBuilder &guardBuilder,
                       llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueVal = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
  });

  burstRead->body([&](mlir::OpBuilder &bodyBuilder,
                      llvm::ArrayRef<mlir::BlockArgument> args) {
    auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());

    auto elementSizeConst =
        UInt::constant(entryInfo.dataWidth / 8, 64, bodyBuilder,
                       wrapper->getLoc()); // Element size in bytes
    auto numBanksConst =
        UInt::constant(entryInfo.numBanks, 64, bodyBuilder, wrapper->getLoc());
    auto myBankConst =
        UInt::constant(bankIdx, 64, bodyBuilder, wrapper->getLoc());
    auto elementsPerBurstConst =
        UInt::constant(elementsPerBurst, 64, bodyBuilder, wrapper->getLoc());

    // element_idx = addr / element_size
    auto elementIdx = addr / elementSizeConst;
    // start_bank_idx = element_idx % num_banks
    auto startBankIdx = elementIdx % numBanksConst;

    // Check if this bank participates: participatesInBurst = (position <
    // elements_per_burst) position = (my_bank - start_bank_idx + num_banks) %
    // num_banks
    auto position =
        (myBankConst - startBankIdx + numBanksConst) % numBanksConst;
    auto isMine = position < elementsPerBurstConst;

    // Calculate local address: my_element_idx = element_idx + position;
    // local_addr = my_element_idx / num_banks
    auto myElementIdx = elementIdx + position;
    auto localAddr = myElementIdx / numBanksConst;
    auto localAddrTrunc = localAddr.bits(entryInfo.addrWidth - 1, 0);

    auto bankDataValues =
        bankInst->callMethod("read", {localAddrTrunc.getValue()}, bodyBuilder);
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

      for (uint32_t elemOffset = 0; elemOffset < elementsPerBurst;
           ++elemOffset) {
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
        auto offsetConst = UInt::constant(i, 64, bodyBuilder, loc);
        auto isThisOffset = elementOffsetInBurst == offsetConst;
        alignedData =
            isThisOffset
                .mux(Signal(positionedDataValues[i], &bodyBuilder, loc),
                     Signal(alignedData, &bodyBuilder, loc))
                .getValue();
      }
    }

    // Return aligned data if mine, else 0
    auto zeroData = UInt::constant(0, 64, bodyBuilder, wrapper->getLoc());
    auto resultOp = isMine.mux(
        Signal(alignedData, &bodyBuilder, wrapper->getLoc()), zeroData);

    bodyBuilder.create<circt::cmt2::ReturnOp>(
        loc, mlir::ValueRange{resultOp.getValue()});
  });

  burstRead->finalize();

  // burst_write method
  auto *burstWrite = wrapper->addMethod(
      "burst_write", {{"addr", u64Type}, {"data", u64Type}}, {});

  burstWrite->guard([&](mlir::OpBuilder &guardBuilder,
                        llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, loc);
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  burstWrite->body([&](mlir::OpBuilder &bodyBuilder,
                       llvm::ArrayRef<mlir::BlockArgument> args) {
    auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());
    auto data = Signal(args[1], &bodyBuilder, wrapper->getLoc());

    // Helper: Create constants using Signal
    auto elementSizeConst =
        UInt::constant(entryInfo.dataWidth / 8, 64, bodyBuilder,
                       wrapper->getLoc()); // Element size in bytes
    auto numBanksConst =
        UInt::constant(entryInfo.numBanks, 64, bodyBuilder, wrapper->getLoc());
    auto myBankConst =
        UInt::constant(bankIdx, 64, bodyBuilder, wrapper->getLoc());
    auto elementsPerBurstConst =
        UInt::constant(elementsPerBurst, 64, bodyBuilder, wrapper->getLoc());

    // Helper 1: element_idx = addr / element_size
    auto elementIdx = addr / elementSizeConst;

    // Helper 2: start_bank_idx = element_idx % num_banks
    auto startBankIdx = elementIdx % numBanksConst;

    // Helper 3: Calculate position and check participation
    auto position =
        ((myBankConst - startBankIdx + numBanksConst) % numBanksConst);
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
    writeAddrWire->callMethod("write", {localAddrTrunc.getValue()},
                              bodyBuilder);

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });

  burstWrite->finalize();

  // Direct bank-level read method (no burst translation), exposes native port
  auto bankAddrType = UIntType::get(builder.getContext(), entryInfo.addrWidth);
  auto bankDataType = UIntType::get(builder.getContext(), entryInfo.dataWidth);

  auto *directReadMethod =
      wrapper->addMethod("bank_read", {{"addr", bankAddrType}}, {bankDataType});

  directReadMethod->guard([&](mlir::OpBuilder &guardBuilder,
                              llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  directReadMethod->body(
      [&, bankInst](mlir::OpBuilder &bodyBuilder,
                    llvm::ArrayRef<mlir::BlockArgument> args) {
        auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());

        // Directly drive the memory bank read ports.
        auto readValues =
            bankInst->callMethod("read", {addr.getValue()}, bodyBuilder);
        bodyBuilder.create<circt::cmt2::ReturnOp>(loc, readValues[0]);
      });

  directReadMethod->finalize();

  auto *directWriteMethod = wrapper->addMethod(
      "bank_write", {{"addr", bankAddrType}, {"data", bankDataType}}, {});

  directWriteMethod->guard([&](mlir::OpBuilder &guardBuilder,
                               llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  directWriteMethod->body(
      [&, bankInst](mlir::OpBuilder &bodyBuilder,
                    llvm::ArrayRef<mlir::BlockArgument> args) {
        auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());
        auto data = Signal(args[1], &bodyBuilder, wrapper->getLoc());

        // Directly drive the memory bank write ports.
        bankInst->callMethod("write", {data.getValue(), addr.getValue()},
                             bodyBuilder);
        bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
      });

  directWriteMethod->finalize();

  // Create a rule that reads from wires and conditionally writes to bank
  // NOTE: This currently crashes because callValue on regular CMT2 Module
  // instances returns empty results. This is a bug in ECMT2 Instance.cpp that
  // needs to be fixed.
  auto *writeRule = wrapper->addRule("do_bank_write");

  writeRule->guard([&](mlir::OpBuilder &guardBuilder) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
    auto enableValues =
        Signal(writeEnableWire->callValue("read", guardBuilder)[0],
               &guardBuilder, loc);
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
  wrapper->setPrecedence(
      {{"burst_read", "bank_read"}, {"burst_write", "bank_write"}});

  return wrapper;
}

// ============================================================================
//
//                   MEMORY ENTRY (CONTAIN BANKWRAPPER)
//
// ============================================================================

/// Generate memory entry submodule for a single memory entry
Module *APSToCMT2GenPass::generateMemoryEntryModule(
    const MemoryEntryInfo &entryInfo, Circuit &circuit, Clock clk, Reset rst,
    const llvm::SmallVector<std::string, 4> &bankNames) {
  // Validate configuration to avoid bank conflicts in burst access
  // Constraint: num_banks * data_width >= 64
  // If < 64, multiple elements in a 64-bit burst map to the same bank
  // (conflict!)
  if (entryInfo.isCyclic) {
    uint32_t totalBankWidth = entryInfo.numBanks * entryInfo.dataWidth;
    if (totalBankWidth < 64) {
      llvm::errs()
          << "  ERROR: Cyclic partition configuration causes bank conflicts!\n";
      llvm::errs() << "    Entry: " << entryInfo.name << "\n";
      llvm::errs() << "    Config: " << entryInfo.numBanks << " banks × "
                   << entryInfo.dataWidth << " bits = " << totalBankWidth
                   << " bits\n";
      llvm::errs() << "    A 64-bit burst contains "
                   << (64 / entryInfo.dataWidth) << " elements\n";
      llvm::errs() << "    Elements per bank: "
                   << ((64 / entryInfo.dataWidth + entryInfo.numBanks - 1) /
                       entryInfo.numBanks)
                   << " (CONFLICT!)\n";
      llvm::errs() << "    Requirement: num_banks × data_width >= 64\n";
      llvm::errs()
          << "    Valid examples: 8×8, 4×16, 2×32, 1×64, 4×32, 8×16, etc.\n";
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

  // Create external module declaration at circuit level (BEFORE creating
  // instances) Save insertion point first
  auto savedIP = builder.saveInsertionPoint();

  llvm::StringMap<int64_t> memParams;
  memParams["data_width"] = entryInfo.dataWidth;
  memParams["addr_width"] = entryInfo.addrWidth;
  memParams["depth"] = entryInfo.depth;

  auto memMod = STLLibrary::createMem1r1w0cModule(
      entryInfo.dataWidth, entryInfo.addrWidth, entryInfo.depth, circuit);

  // Restore insertion point back to entry module
  builder.restoreInsertionPoint(savedIP);

  // Create bank wrapper modules (this changes insertion point to each wrapper)
  llvm::SmallVector<Module *, 4> wrapperModules;
  for (size_t i = 0; i < entryInfo.numBanks; ++i) {
    auto *wrapperMod = generateBankWrapperModule(entryInfo, circuit, i, memMod,
                                                 subClk, subRst);
    wrapperModules.push_back(wrapperMod);
  }

  // Restore insertion point back to entryModule before adding instances/methods
  builder.restoreInsertionPoint(savedIP);

  // Create wrapper instances IN THIS SUBMODULE
  llvm::SmallVector<Instance *, 4> wrapperInstances;
  for (size_t i = 0; i < entryInfo.numBanks; ++i) {
    std::string wrapperName = "bank_wrap_" + std::to_string(i);
    auto *wrapperInst = entryModule->addInstance(
        wrapperName, wrapperModules[i], {subClk.getValue(), subRst.getValue()});
    wrapperInstances.push_back(wrapperInst);
  }

  // Create burst_read method: forwards addr to all banks, ORs results
  auto *burstRead =
      entryModule->addMethod("burst_read", {{"addr", u64Type}}, {u64Type});

  // Guard: always ready (address range checked at parent level)
  burstRead->guard([&](mlir::OpBuilder &guardBuilder,
                       llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  // Body: Simple OR aggregation of all bank wrapper outputs
  // Each wrapper returns aligned 64-bit data if address is for that bank, else
  // 0
  burstRead->body([&](mlir::OpBuilder &bodyBuilder,
                      llvm::ArrayRef<mlir::BlockArgument> args) {
    auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());

    // Use CallOp to call wrapper methods (Instance::callMethod doesn't work for
    // methods with return values)
    auto calleeSymbol0 = mlir::FlatSymbolRefAttr::get(
        bodyBuilder.getContext(), wrapperInstances[0]->getName());
    auto methodSymbol =
        mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_read");
    auto callOp0 = bodyBuilder.create<circt::cmt2::CallOp>(
        loc, mlir::TypeRange{u64Type}, mlir::ValueRange{addr.getValue()},
        calleeSymbol0, methodSymbol, nullptr, nullptr);
    auto result =
        Signal(callOp0.getResult(0), &bodyBuilder, entryModule->getLoc());

    // OR together all other wrapper outputs
    for (size_t i = 1; i < entryInfo.numBanks; ++i) {
      auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
          bodyBuilder.getContext(), wrapperInstances[i]->getName());
      auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{u64Type}, mlir::ValueRange{addr.getValue()},
          calleeSymbol, methodSymbol, nullptr, nullptr);
      auto data =
          Signal(callOp.getResult(0), &bodyBuilder, entryModule->getLoc());
      result = result | data;
    }

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc, result.getValue());
  });

  burstRead->finalize();
  // Create burst_write method: forwards data to all banks
  auto *burstWrite = entryModule->addMethod(
      "burst_write", {{"addr", u64Type}, {"data", u64Type}}, {});

  burstWrite->guard([&](mlir::OpBuilder &guardBuilder,
                        llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  burstWrite->body([&](mlir::OpBuilder &bodyBuilder,
                       llvm::ArrayRef<mlir::BlockArgument> args) {
    auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());
    auto data = Signal(args[1], &bodyBuilder, entryModule->getLoc());

    // Simple broadcast to all bank wrappers using CallOp
    // Each wrapper decides if it should write based on address
    auto methodSymbol =
        mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_write");
    for (size_t i = 0; i < entryInfo.numBanks; ++i) {
      auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
          bodyBuilder.getContext(), wrapperInstances[i]->getName());
      bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{},
          mlir::ValueRange{addr.getValue(), data.getValue()}, calleeSymbol,
          methodSymbol, nullptr, nullptr);
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
        bankReadName, {{"addr", bankAddrType}}, {bankDataType});

    bankReadMethod->guard([&](mlir::OpBuilder &guardBuilder,
                              llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
      auto trueConst =
          UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    bankReadMethod->body([&,
                          bankIdx](mlir::OpBuilder &bodyBuilder,
                                   llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());
      auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
          bodyBuilder.getContext(), wrapperInstances[bankIdx]->getName());
      auto methodSymbol =
          mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "bank_read");
      auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{bankDataType}, mlir::ValueRange{addr.getValue()},
          calleeSymbol, methodSymbol, bodyBuilder.getArrayAttr({}),
          bodyBuilder.getArrayAttr({}));
      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, callOp.getResult(0));
    });

    bankReadMethod->finalize();

    auto *bankWriteMethod = entryModule->addMethod(
        bankWriteName, {{"addr", bankAddrType}, {"data", bankDataType}}, {});

    bankWriteMethod->guard([&](mlir::OpBuilder &guardBuilder,
                               llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
      auto trueConst =
          UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    bankWriteMethod->body([&,
                           bankIdx](mlir::OpBuilder &bodyBuilder,
                                    llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());
      auto data = Signal(args[1], &bodyBuilder, entryModule->getLoc());
      auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
          bodyBuilder.getContext(), wrapperInstances[bankIdx]->getName());
      auto methodSymbol =
          mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "bank_write");
      bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{},
          mlir::ValueRange{addr.getValue(), data.getValue()}, calleeSymbol,
          methodSymbol, bodyBuilder.getArrayAttr({}),
          bodyBuilder.getArrayAttr({}));
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
void APSToCMT2GenPass::generateBurstAccessLogic(
    Module *poolModule, const llvm::SmallVector<MemoryEntryInfo> &memEntryInfos,
    Circuit &circuit, Clock clk, Reset rst) {

  auto &builder = poolModule->getBuilder();
  auto loc = poolModule->getLoc();
  auto u64Type = UIntType::get(builder.getContext(), 64);

  // Step 1: Create all memory entry submodules
  // Save the insertion point first, since circuit.addModule() will change it
  auto savedIP = builder.saveInsertionPoint();

  llvm::SmallVector<Module *, 4> entryModules;
  for (const auto &entryInfo : memEntryInfos) {
    // Collect bank names
    llvm::SmallVector<std::string, 4> bankNames;
    for (size_t i = 0; i < entryInfo.numBanks; ++i) {
      bankNames.push_back(entryInfo.name + "_" + std::to_string(i));
    }

    // Create submodule for this memory entry (this will change insertion point)
    auto *entryMod =
        generateMemoryEntryModule(entryInfo, circuit, clk, rst, bankNames);
    entryModules.push_back(entryMod);
  }

  // Step 2: Restore insertion point back to poolModule's body before creating
  // instances
  builder.restoreInsertionPoint(savedIP);

  llvm::SmallVector<Instance *, 4> entryInstances;
  for (size_t i = 0; i < memEntryInfos.size(); ++i) {
    const auto &entryInfo = memEntryInfos[i];
    auto *entryMod = entryModules[i];

    std::string instanceName = "inst_" + entryInfo.name;

    auto *entryInst = poolModule->addInstance(instanceName, entryMod,
                                              {clk.getValue(), rst.getValue()});
    entryInstances.push_back(entryInst);
  }

  // Create top-level burst_read method that dispatches to correct memory entry
  auto *topBurstRead =
      poolModule->addMethod("burst_read", {{"addr", u64Type}}, {u64Type});

  topBurstRead->guard([&](mlir::OpBuilder &guardBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  topBurstRead->body([&](mlir::OpBuilder &bodyBuilder,
                         llvm::ArrayRef<mlir::BlockArgument> args) {
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

      // Call submodule's burst_read (manual CallOp needed for methods with
      // return values)
      auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(),
                                                       entryInst->getName());
      auto methodSymbol =
          mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_read");
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
      "burst_write", {{"addr", u64Type}, {"data", u64Type}}, {}
      // No return value
  );

  topBurstWrite->guard([&](mlir::OpBuilder &guardBuilder,
                           llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  topBurstWrite->body([&](mlir::OpBuilder &bodyBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
    auto addr = Signal(args[0], &bodyBuilder, poolModule->getLoc());
    auto data = Signal(args[1], &bodyBuilder, poolModule->getLoc());

    // Dispatch to all memory entries based on address range
    for (size_t i = 0; i < memEntryInfos.size(); ++i) {
      const auto &entryInfo = memEntryInfos[i];
      auto *entryInst = entryInstances[i];

      uint32_t entryStart = entryInfo.baseAddress;
      uint32_t entryEnd = entryStart + entryInfo.bankSize;

      // Calculate relative address for this entry
      auto startConst =
          UInt::constant(entryStart, 64, bodyBuilder, poolModule->getLoc());
      auto endConst =
          UInt::constant(entryEnd, 64, bodyBuilder, poolModule->getLoc());

      auto relAddr = (addr - startConst).bits(63, 0);
      auto inRange = (addr >= startConst) & (addr < endConst);

      // Only call submodule's burst_write if address is in range
      // Use If to conditionally execute the call
      If(
          inRange,
          [&](mlir::OpBuilder &thenBuilder) {
            auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
                thenBuilder.getContext(), entryInst->getName());
            auto methodSymbol = mlir::FlatSymbolRefAttr::get(
                thenBuilder.getContext(), "burst_write");
            thenBuilder.create<circt::cmt2::CallOp>(
                loc, mlir::TypeRange{},
                mlir::ValueRange{relAddr.getValue(), data.getValue()},
                calleeSymbol, methodSymbol, nullptr, nullptr);
          },
          bodyBuilder, loc);
    }

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });

  topBurstWrite->finalize();

  // Expose direct per-bank read/write methods at the top level (no burst
  // translation).
  for (size_t entryIdx = 0; entryIdx < memEntryInfos.size(); ++entryIdx) {
    const auto &entryInfo = memEntryInfos[entryIdx];
    auto bankAddrType =
        UIntType::get(builder.getContext(), entryInfo.addrWidth);
    auto bankDataType =
        UIntType::get(builder.getContext(), entryInfo.dataWidth);

    for (size_t bankIdx = 0; bankIdx < entryInfo.numBanks; ++bankIdx) {
      std::string topReadName =
          entryInfo.name + "_bank" + std::to_string(bankIdx) + "_read";
      std::string topWriteName =
          entryInfo.name + "_bank" + std::to_string(bankIdx) + "_write";
      std::string entryReadName = "bank_read_" + std::to_string(bankIdx);
      std::string entryWriteName = "bank_write_" + std::to_string(bankIdx);

      auto *topBankRead = poolModule->addMethod(
          topReadName, {{"addr", bankAddrType}}, {bankDataType});

      topBankRead->guard([&](mlir::OpBuilder &guardBuilder,
                             llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
        auto trueConst =
            UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
        guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
      });

      topBankRead->body([&, entryIdx, entryReadName](
                            mlir::OpBuilder &bodyBuilder,
                            llvm::ArrayRef<mlir::BlockArgument> args) {
        auto addr = Signal(args[0], &bodyBuilder, poolModule->getLoc());
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
            bodyBuilder.getContext(), entryInstances[entryIdx]->getName());
        auto methodSymbol = mlir::FlatSymbolRefAttr::get(
            bodyBuilder.getContext(), entryReadName);
        auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
            loc, mlir::TypeRange{bankDataType},
            mlir::ValueRange{addr.getValue()}, calleeSymbol, methodSymbol,
            bodyBuilder.getArrayAttr({}), bodyBuilder.getArrayAttr({}));
        bodyBuilder.create<circt::cmt2::ReturnOp>(loc, callOp.getResult(0));
      });

      topBankRead->finalize();

      auto *topBankWrite = poolModule->addMethod(
          topWriteName, {{"addr", bankAddrType}, {"data", bankDataType}}, {});

      topBankWrite->guard([&](mlir::OpBuilder &guardBuilder,
                              llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
        auto trueConst =
            UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
        guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
      });

      topBankWrite->body([&, entryIdx, entryWriteName](
                             mlir::OpBuilder &bodyBuilder,
                             llvm::ArrayRef<mlir::BlockArgument> args) {
        auto addr = Signal(args[0], &bodyBuilder, poolModule->getLoc());
        auto data = Signal(args[1], &bodyBuilder, poolModule->getLoc());
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
            bodyBuilder.getContext(), entryInstances[entryIdx]->getName());
        auto methodSymbol = mlir::FlatSymbolRefAttr::get(
            bodyBuilder.getContext(), entryWriteName);
        bodyBuilder.create<circt::cmt2::CallOp>(
            loc, mlir::TypeRange{},
            mlir::ValueRange{addr.getValue(), data.getValue()}, calleeSymbol,
            methodSymbol, bodyBuilder.getArrayAttr({}),
            bodyBuilder.getArrayAttr({}));
        bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
      });

      topBankWrite->finalize();
    }
  }
}

void APSToCMT2GenPass::addRoCCAndMemoryMethodToMainModule(
    Module *mainModule, Instance *roccInstance, Instance *hellaMemInstance) {
  auto &builder = mainModule->getBuilder();
  auto roccCmdBundleType = getRoccCmdBundleType(builder);
  auto hellaRespBundleType = getHellaRespBundleType(builder);
  auto loc = mainModule->getLoc();
  auto *roccCmdMethod = mainModule->addMethod(
      "rocc_cmd", {{{"rocc_cmd", roccCmdBundleType}}}, {});

  roccCmdMethod->guard([&](mlir::OpBuilder &guardBuilder,
                           llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
  });
  roccCmdMethod->body([&](mlir::OpBuilder &bodyBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
    auto arg = args[0];
    auto loc = bodyBuilder.getUnknownLoc();
    roccInstance->callMethod("cmd_from_bus", arg, bodyBuilder);
    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });
  roccCmdMethod->finalize();

  auto *hellaRespMethod = mainModule->addMethod(
      "hella_resp", {{"hella_resp", hellaRespBundleType}}, {});

  hellaRespMethod->guard([&](mlir::OpBuilder &guardBuilder,
                             llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
  });
  hellaRespMethod->body([&](mlir::OpBuilder &bodyBuilder,
                            llvm::ArrayRef<mlir::BlockArgument> args) {
    auto arg = args[0];
    auto loc = bodyBuilder.getUnknownLoc();
    hellaMemInstance->callMethod("resp_from_bus", arg, bodyBuilder);
    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });
  hellaRespMethod->finalize();
}

/// Get memref.global by symbol name
memref::GlobalOp APSToCMT2GenPass::getGlobalMemRef(mlir::Operation *scope,
                                                   StringRef symbolName) {
  if (!scope)
    return nullptr;
  auto symRef = mlir::FlatSymbolRefAttr::get(scope->getContext(), symbolName);
  if (auto *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(scope, symRef))
    return mlir::dyn_cast<memref::GlobalOp>(symbol);
  return nullptr;
}

/// Generate the CMT2 memory pool module
MemoryPoolResult
APSToCMT2GenPass::generateMemoryPool(Circuit &circuit, ModuleOp moduleOp,
                                     aps::MemoryMapOp memoryMapOp) {
  llvm::outs() << "DEBUG: generateMemoryPool() started\n";
  MLIRContext *context = moduleOp.getContext();
  OpBuilder builder(context);

  // Collect all memory entries
  llvm::SmallVector<aps::MemEntryOp> memEntries;
  llvm::outs() << "DEBUG: About to collect memory entries\n";

  if (memoryMapOp) {
    llvm::outs() << "DEBUG: Memory map region has "
                 << memoryMapOp.getRegion().getBlocks().size() << " blocks\n";

    // Safely iterate through operations
    for (auto &block : memoryMapOp.getRegion()) {
      for (auto &op : block) {
        if (auto entry = dyn_cast<aps::MemEntryOp>(op)) {
          memEntries.push_back(entry);
          llvm::outs() << "DEBUG: Found memory entry: " << entry.getName()
                       << "\n";
        }
      }
    }

    llvm::outs() << "DEBUG: Collected " << memEntries.size()
                 << " memory entries\n";
  }

  if (memEntries.empty()) {
    llvm::outs() << "DEBUG: No memory entries found, will generate rules "
                    "without memory\n";
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
    auto globalOp =
        getGlobalMemRef(memEntry.getOperation(), firstBankSymAttr.getValue());
    if (!globalOp) {
      llvm::errs() << "Error: Could not find memref.global for bank "
                   << firstBankSymAttr.getValue() << "\n";
      continue;
    }

    // Extract memory parameters from the memref type
    int dataWidth = 32; // defaults
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

} // namespace mlir