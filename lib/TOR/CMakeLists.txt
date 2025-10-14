add_mlir_dialect_library(MLIRTOR
    TORDialect.cpp
    TOR.cpp
    TORTypes.cpp
    SCFDump.cpp
    # FuncMemrefExtract.cpp
    SCFIterArgs.cpp
    SCFToTOR.cpp
    MAxiBurstInfer.cpp
    TORSchedulePass.cpp
    TORSplitPass.cpp
    TORDump.cpp
    # ConvertStream.cpp
    ConvertInput.cpp
    TORAttrs.cpp
    ArrayPartition.cpp
    SCEVAnalysis.cpp
    DependenceAnalysis.cpp

    RemoveRedundantAccess.cpp
    # KrnlLower.cpp
    UnificationIndexCast.cpp
    # FuseLoop.cpp

    LowerAffineFor.cpp
    # ExtractStridedMetadataLower.cpp
    RemoveFuncArgPass.cpp
    ReinterpretCastLower.cpp
    MinMaxToCmpSelectPass.cpp
    # LinalgConvToAffineLower.cpp
    HlsUnroll.cpp
    IgnoreAliasingLICM.cpp
    # Mem2IterArgs.cpp
    NewArrayPartition.cpp
    CountCycles.cpp
    GenPragmaReport.cpp
    AddPragma.cpp
    # StructSplit.cpp
    ArrayOpt.cpp
    DemangleFuncName.cpp
    # ONNXLower.cpp
    ConvertMathToCall.cpp
    # TopFuncArg2Global.cpp
    # ArrayUseOffset.cpp

    TORFuse.cpp
    # LoopFlatten.cpp
    # LoopMerge.cpp
    LoopTripcount.cpp
    ExpressionBalance.cpp
    MemrefReuse.cpp
    MemrefGlobalToConstant.cpp
    EmbedWeightsAndOptimizeLoadPass.cpp
    # TosaReshapeNoDataLayoutPass.cpp

    RaiseToAffine.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TOR
    ${RROJECT_SOURCE_DIR}/include/Schedule
    DEPENDS
    MLIRTORIncGen
    MLIRTOREnumsIncGen
    MLIRTORPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
