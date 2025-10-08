#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "TOR/TORDialect.h"
#include "TOR/Passes.h"

#include <sstream>
#include <stdlib.h>
#include <string>
#include <cstring>
#include "fstream"
#include "iostream"
#include "sys/io.h"
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <filesystem>

#include "hector-driver.cc"
#include "argparse.hpp"

int main(int argc, char **argv) {
    argparse::ArgumentParser program("hector-lite", "0.1");
    program.add_argument("-i", "--input").required()
            .help("Input MLIR file (e.g., examples/zyy.mlir)");
    program.add_argument("-f", "--function").required()
            .help("Specify the target function");
    program.add_argument("-o", "--output").required()
            .help("Output path (e.g., zyy_out)");
    program.add_argument("--clock").required().default_value(std::string("10.0"))
            .help("Clock period in ns (default: 10.0)");
    program.add_argument("--resource").required().default_value(std::string("examples/resource.json"))
            .help("Path to resource.json for scheduling information");
    program.add_argument("--gensg").default_value(false).implicit_value(true)
            .help("Generate schedule graph and estimate the running cycle");
    program.add_argument("--genpr").default_value(false).implicit_value(true)
            .help("Generate pragma report");
    program.add_argument("--genrr").default_value(false).implicit_value(true)
            .help("Generate resource report");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto arg_input = program.get<std::string>("--input");
    auto arg_clock = program.get<std::string>("--clock");
    auto arg_resource = program.get<std::string>("--resource");
    auto arg_output = program.get<std::string>("--output");
    auto arg_gensg = program.get<bool>("--gensg");
    auto arg_genpr = program.get<bool>("--genpr");
    auto arg_genrr = program.get<bool>("--genrr");

    if (access(arg_input.c_str(), F_OK) != 0) {
        std::cerr << "[ERROR] " << arg_input << " is not exist" << std::endl;
        return 0;
    }

    if (access(arg_resource.c_str(), F_OK) != 0) {
        std::cerr << "[ERROR] Resource file " << arg_resource << " does not exist" << std::endl;
        return 1;
    }

    std::stringstream ss;

    /////////// hector ///////////
    int hector_argc = 50;

    if (arg_gensg) {
        hector_argc += 2;
    }
    if (arg_genpr) {
        hector_argc += 1;
    }
    if (arg_genrr) {
        hector_argc += 1;
    }

    char **hector_argv;
    hector_argv = (char **) malloc(sizeof(char *) * hector_argc);

    int index = 0;
    hector_argv[index++] = argv[0];
    // 输入文件
    hector_argv[index++] = const_cast<char *>(arg_input.c_str());

    hector_argv[index++] = const_cast<char *>("--allow-unregistered-dialect");
    hector_argv[index++] = const_cast<char *>("--mlir-print-ir-after-all");
    // hector_argv[index++] = const_cast<char *>("--mlir-print-op-generic");

    ss << "--convert-math-to-call=resource=" << arg_resource.c_str();
    hector_argv[index] = static_cast<char *>(malloc(strlen(ss.str().c_str()) + 1));
    strcpy(hector_argv[index++], ss.str().c_str());
    ss.clear();
    ss.str("");

    hector_argv[index++] = const_cast<char *>("--demangle-func-name");

    // hector_argv[index++] = const_cast<char *>("--canonicalize");
    // hector_argv[index++] = const_cast<char *>("--struct-split");
    // hector_argv[index++] = const_cast<char *>("--canonicalize");
    // hector_argv[index++] = const_cast<char *>("--unification-index-cast");
    // hector_argv[index++] = const_cast<char *>("--hls-unroll");
    // hector_argv[index++] = const_cast<char *>("--canonicalize");
    // hector_argv[index++] = const_cast<char *>("--affine-loop-normalize");
    // hector_argv[index++] = const_cast<char *>("--canonicalize");
    // hector_argv[index++] = const_cast<char *>("--new-array-partition");
    // hector_argv[index++] = const_cast<char *>("--canonicalize");
    // hector_argv[index++] = const_cast<char *>("--array-opt");
    // hector_argv[index++] = const_cast<char *>("--loop-merge");
    if (arg_gensg) {
      hector_argv[index++] = const_cast<char *>("--loop-tripcount");
    }
    // hector_argv[index++] = const_cast<char *>("--canonicalize");

    hector_argv[index++] = const_cast<char *>("--canonicalize");
    // hector_argv[index++] = const_cast<char *>("--detect-reduction");

    // ss << "--attribute-deletion=output-file=" << arg_output.c_str();
    // hector_argv[index] = static_cast<char *>(malloc(strlen(ss.str().c_str()) + 1));
    // strcpy(hector_argv[index++], ss.str().c_str()); ss.clear(); ss.str("");

    // affine to scf
    // hector_argv[index++] = const_cast<char *>("--lower-affine-for");
    // hector_argv[index++] = const_cast<char *>("--canonicalize");
    // hector_argv[index++] = const_cast<char *>("--array-use-offset");
    // hector_argv[index++] = const_cast<char *>("--canonicalize");
    // hector_argv[index++] = const_cast<char *>("--mem-to-iter-args");
    // hector_argv[index++] = const_cast<char *>("--loop-flatten");
    // hector_argv[index++] = const_cast<char *>("--canonicalize");
    hector_argv[index++] = const_cast<char *>("--expression-balance");
    // convert input
    
    auto arg_function = program.get<std::string>("--function");
    ss << "--convert-input=top-function=" << arg_function.c_str() << " clock=" << arg_clock.c_str() << " resource="
    << arg_resource.c_str();
    ss << " output-path=" << arg_output.c_str() << "/file";
    hector_argv[index] = static_cast<char *>(malloc(strlen(ss.str().c_str()) + 1));
    strcpy(hector_argv[index++], ss.str().c_str());
    ss.clear();
    ss.str("");

    // hector_argv[index++] = const_cast<char *>("--canonicalize");
    // hector_argv[index++] = const_cast<char *>("--func-extract");
    // hector_argv[index++] = const_cast<char *>("--scf-iterargs");
    // dump-scf
    ss << "--dump-scf=json=" << arg_output.c_str() << "/scf.json";
    hector_argv[index] = static_cast<char *>(malloc(strlen(ss.str().c_str()) + 1));
    strcpy(hector_argv[index++], ss.str().c_str()); ss.clear(); ss.str("");

    hector_argv[index++] = const_cast<char *>("--canonicalize");
    hector_argv[index++] = const_cast<char *>("--scf-to-tor");
    hector_argv[index++] = const_cast<char *>("--canonicalize");
    hector_argv[index++] = const_cast<char *>("--maxi-burst-infer");
    hector_argv[index++] = const_cast<char *>("--canonicalize");
    hector_argv[index++] = const_cast<char *>("--schedule-tor");
    hector_argv[index++] = const_cast<char *>("--canonicalize");
    hector_argv[index++] = const_cast<char *>("--split-schedule");
    // dump-tor
    // ss << "--dump-tor=json=" << arg_output.c_str() << "/tor.json";
    // hector_argv[index] = static_cast<char *>(malloc(strlen(ss.str().c_str()) + 1));
    // strcpy(hector_argv[index++], ss.str().c_str()); ss.clear(); ss.str("");

    if (arg_genpr) {
        hector_argv[index++] = const_cast<char *>("--generate-pragma-report");
    }

    if (arg_gensg) {
        // 输出timegraph
        ss << "--count-cycles=output-dir=" << arg_output.c_str();
        hector_argv[index] = static_cast<char *>(malloc(strlen(ss.str().c_str()) + 1));
        strcpy(hector_argv[index++], ss.str().c_str());
        ss.clear();
        ss.str("");
    }

    hector_driver(index, hector_argv);

    return 0;
}
