#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "TOR/DialectCreater.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <fstream>
#include <ios>
#include <list>
#include <string>
#define DEBUG_TYPE "embed-weights-optimize-load"


namespace {
    using namespace mlir;
    struct EmbedWeightsPattern : public OpConversionPattern<tensor::ExtractSliceOp> {
        using OpConversionPattern::OpConversionPattern;
        std::string data_path;
        EmbedWeightsPattern(MLIRContext *ctx, std::string data_path) :
            OpConversionPattern<tensor::ExtractSliceOp>(ctx), data_path(data_path) {}

        
        LogicalResult matchAndRewrite(tensor::ExtractSliceOp tensorSliceOp, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
            auto loc = tensorSliceOp.getLoc();
            auto start = tensorSliceOp.getStaticOffsets();
            auto sizes = tensorSliceOp.getStaticSizes();
            auto outputType = dyn_cast<ShapedType>(tensorSliceOp->getResultTypes().front());
            std::string dataFilePath = data_path + "/hex_mem_" +
                                       std::to_string(start[0]) + "_" +
                                       std::to_string(sizes[0]) + ".txt";
            std::ifstream dataFile(dataFilePath);
            std::string line;
            std::vector<float> data;
            while (std::getline(dataFile, line)) {
                line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
                line.erase(std::remove(line.begin(), line.end(), ' '), line.end());

                unsigned int hexValue;
                std::istringstream(line) >> std::hex >> hexValue;
                float floatValue;
                std::memcpy(&floatValue, &hexValue, sizeof(float));
                data.push_back(floatValue);
            }
            
            SmallVector<Attribute, 4> floatAttrs;
            for (float value : data) {
                floatAttrs.push_back(rewriter.getF32FloatAttr(value));
            }

            auto tensorAttr = DenseElementsAttr::get(outputType, floatAttrs);
            auto constOp =rewriter.create<arith::ConstantOp>(loc, tensorAttr.getType(), tensorAttr);
            rewriter.replaceOp(tensorSliceOp, constOp);
            return success();
        }
    };
    
    struct EmbedWeightsAndOptimizeLoadPass : public EmbedWeightsAndOptimizeLoadBase<EmbedWeightsAndOptimizeLoadPass> {
        void getDependentDialects(DialectRegistry &registry) const override {
            registry.insert<mlir::tosa::TosaDialect>();
        }
        void runOnOperation() override {
            MLIRContext &ctxt = getContext();
            ConversionTarget target(ctxt);
            target.addLegalDialect<tosa::TosaDialect>();
            target.addLegalOp<arith::ConstantOp>();
            RewritePatternSet patterns(&ctxt);
            patterns.add<EmbedWeightsPattern>(&ctxt, data_path);

            if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
                signalPassFailure();
            }
        }
    };

}; // namespace



namespace mlir {
    std::unique_ptr<Pass> createEmbedWeightsAndOptimizeLoadPass() {
        return std::make_unique<EmbedWeightsAndOptimizeLoadPass>();
    }
} // namespace mlir


