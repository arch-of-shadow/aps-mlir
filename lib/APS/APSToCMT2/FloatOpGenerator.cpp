//===- FloatOpGenerator.cpp - Scheduled BF16/FP32 issue/collect --------------===//
#include "APS/FloatOpGenerator.h"
#include "APS/FloatIPLibrary.h"
#include "APS/HardwareValueUtils.h"
#include "APS/HardFloatConfig.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
namespace mlir {
bool FloatOpGenerator::canHandle(Operation *op) const {
  return isa<aps::FPOp,tor::AddFOp,tor::SubFOp,tor::MulFOp,tor::DivFOp,tor::CmpFOp,
    arith::AddFOp,arith::SubFOp,arith::MulFOp,arith::DivFOp,arith::CmpFOp,
    arith::SIToFPOp,arith::UIToFPOp,arith::FPToSIOp,arith::FPToUIOp,
    arith::ExtFOp,arith::TruncFOp,arith::NegFOp,math::FmaOp,math::SqrtOp>(op);
}
static unsigned kindOf(Operation *op) {
 if(auto fp=dyn_cast<aps::FPOp>(op)) return op->getAttrOfType<IntegerAttr>("kind").getInt();
 if(isa<tor::SubFOp,arith::SubFOp>(op))return 1;
 if(isa<tor::MulFOp,arith::MulFOp>(op))return 2;
 if(isa<tor::DivFOp,arith::DivFOp>(op))return 3;
 if(isa<math::SqrtOp>(op))return 4;
 if(isa<math::FmaOp>(op))return 5;
 if(isa<tor::CmpFOp,arith::CmpFOp>(op))return 6;
 if(isa<arith::SIToFPOp>(op))return 7;
 if(isa<arith::UIToFPOp>(op))return 8;
 if(isa<arith::FPToSIOp>(op))return 9;
 if(isa<arith::FPToUIOp>(op))return 10;
 if(isa<arith::ExtFOp>(op))return 11;
 if(isa<arith::TruncFOp>(op))return 12;
 if(isa<arith::NegFOp>(op))return 13;
 return 0;
}
LogicalResult FloatOpGenerator::prepare(Operation *op) {
  // Validate before creating rule callbacks: their failure path is fatal.
  for (Type type : op->getOperandTypes()) {
    auto bits = getHardwareBitWidth(type);
    if (!bits || *bits > 32)
      return op->emitError("HardFloat operands must be scalar f32/bf16 or integers up to 32 bits");
  }
  for (Type type : op->getResultTypes()) {
    auto bits = getHardwareBitWidth(type);
    if (!bits || *bits > 32)
      return op->emitError("HardFloat results must be scalar f32/bf16 or integers up to 32 bits");
  }
  unsigned kind=kindOf(op),width=32,predicate=0;
  if(auto fp=dyn_cast<aps::FPOp>(op)) {
    width=op->getAttrOfType<IntegerAttr>("width").getInt();
    predicate=op->getAttrOfType<IntegerAttr>("predicate").getInt();
    auto rm=op->getAttrOfType<IntegerAttr>("rm").getInt();
    if(kind>18 || (width!=16&&width!=32) || rm<0||rm>4||predicate>15)
      return op->emitError("invalid HardFloat operation/precision/rounding/predicate");
    unsigned arity = kind==5 ? 3 : (kind<=3 || kind==6 || kind==15 || kind==16 || kind==17) ? 2 : 1;
    if(op->getNumOperands()!=arity || !op->getResult(1).getType().isInteger(5))
      return op->emitError("invalid HardFloat operands/flags type");
    Type precision = width==16 ? Type(BFloat16Type::get(op->getContext())) : Type(Float32Type::get(op->getContext()));
    Type inputType = (kind==7 || kind==8) ? Type(IntegerType::get(op->getContext(),32)) :
      kind==11 ? Type(BFloat16Type::get(op->getContext())) : kind==12 ? Type(Float32Type::get(op->getContext())) : precision;
    Type outputType = kind==6 ? Type(IntegerType::get(op->getContext(),1)) :
      (kind==9 || kind==10 || kind==18) ? Type(IntegerType::get(op->getContext(),32)) :
      kind==11 ? Type(Float32Type::get(op->getContext())) : kind==12 ? Type(BFloat16Type::get(op->getContext())) : precision;
    if(llvm::any_of(op->getOperandTypes(),[&](Type t){return t!=inputType;}) || op->getResult(0).getType()!=outputType)
      return op->emitError("HardFloat semantic operand/result types do not match the operation precision");
  } else {
    Type fpType=op->getResult(0).getType();
    if(!isa<FloatType>(fpType))fpType=op->getOperand(0).getType();
    if(!fpType.isF32()&&!fpType.isBF16())
      return op->emitError("HardFloat backend supports only scalar f32/bf16");
    width=fpType.isBF16()?16:32;
    if(auto cmp=dyn_cast<tor::CmpFOp>(op))predicate=static_cast<unsigned>(cmp.getPredicate());
    if(auto cmp=dyn_cast<arith::CmpFOp>(op))predicate=static_cast<unsigned>(cmp.getPredicate());
  }
  if(auto flags=op->getAttrOfType<arith::FastMathFlagsAttr>("fastmath"))
    if(flags.getValue()!=arith::FastMathFlags::none)
      return op->emitError("HardFloat requires strict floating-point semantics");
  auto start=op->getAttrOfType<IntegerAttr>("starttime");
  auto end=op->getAttrOfType<IntegerAttr>("endtime");
  if(!start||!end||end.getInt()<=start.getInt())
    return op->emitError("floating-point issue/collect requires endtime > starttime; use the generated FP resources");
  unsigned latency=hardfloat::latency(kind,width);
  auto rs=op->getAttrOfType<IntegerAttr>("ref_starttime"),re=op->getAttrOfType<IntegerAttr>("ref_endtime");
  if(!rs || !re || re.getInt()-rs.getInt()!=latency)
    return op->emitError("HardFloat schedule disagrees with native operation latency; regenerate resources and reschedule");
  auto *module=createFloatIPModule(bbHandler->getCircuit(),kind,latency,width,predicate);
  if(!module)return op->emitError("failed to create HardFloat module");
  instances[op]=bbHandler->getMainModule()->addInstance(bbHandler->nextFloatInstanceName(),module,
    {bbHandler->getMainClk().getValue(),bbHandler->getMainRst().getValue()});
  completions[end.getInt()].push_back(op);
  return success();
}
LogicalResult FloatOpGenerator::generateRule(Operation *op,OpBuilder &b,Location loc,
  int64_t slot,llvm::DenseMap<Value,Value> &localMap) {
  llvm::SmallVector<Value> args;
  for(Value operand:op->getOperands()) {
    auto v=getValueInRule(operand,op,b,localMap,loc);
    if(failed(v))return failure();
    Signal s(*v,&b,loc);
    if(s.getWidth()>32)return op->emitError("HardFloat integer operands must fit 32 bits");
    if(kindOf(op)==7&&s.getWidth()<32) {
      auto si=b.create<circt::firrtl::AsSIntPrimOp>(loc,circt::firrtl::SIntType::get(b.getContext(),s.getWidth()),*v);
      auto pad=b.create<circt::firrtl::PadPrimOp>(loc,circt::firrtl::SIntType::get(b.getContext(),32),si,32);
      args.push_back(b.create<circt::firrtl::AsUIntPrimOp>(loc,circt::firrtl::UIntType::get(b.getContext(),32),pad));
    } else args.push_back(s.pad(32).getValue());
  }
  while(args.size()<3)args.push_back(UInt::constant(0,32,b,loc).getValue());
  unsigned rm=isa<arith::FPToSIOp,arith::FPToUIOp>(op)?1:0;
  if(isa<aps::FPOp>(op))rm=op->getAttrOfType<IntegerAttr>("rm").getInt();
  args.push_back(UInt::constant(rm,3,b,loc).getValue());
  instances.lookup(op)->callMethod("issue",args,b);
  return success();
}
LogicalResult FloatOpGenerator::collectResults(int64_t slot,OpBuilder &b,
  llvm::DenseMap<Value,Value> &localMap) {
  auto it=completions.find(slot);if(it==completions.end())return success();
  for(Operation *op:it->second) {
    auto r=instances.lookup(op)->callMethod("collect",{},b);
    if(r.size()!=2)return op->emitError("HardFloat collect must return data and flags");
    auto w=getHardwareBitWidth(op->getResult(0).getType());
    if(!w||*w>32)return op->emitError("unsupported HardFloat result type");
    localMap[op->getResult(0)]=Signal(r[0],&b,op->getLoc()).bits(*w-1,0).getValue();
    if(op->getNumResults()==2)localMap[op->getResult(1)]=r[1];
  }
  return success();
}
llvm::SmallVector<Value> FloatOpGenerator::getResults(int64_t slot) const {
 llvm::SmallVector<Value> r;auto it=completions.find(slot);
 if(it!=completions.end())for(Operation *op:it->second)llvm::append_range(r,op->getResults());
 return r;
}
} // namespace mlir
