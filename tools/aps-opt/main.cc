#include "aps_opt.hpp"
#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <climits>

// Helper function to execute a command and capture stdout/stderr
std::pair<std::string, std::string> executeCommand(const std::string &command) {
    std::array<char, 128> buffer;
    std::string stdout_result;
    std::string stderr_result;

    // Redirect stderr to a temporary file
    std::string tempErrorFile = "/tmp/aps_opt_stderr_" + std::to_string(getpid()) + ".txt";
    std::string fullCommand = command + " 2> " + tempErrorFile;

    // Execute command and capture stdout
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(fullCommand.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        stdout_result += buffer.data();
    }

    // Read stderr from temporary file
    std::ifstream errorFile(tempErrorFile);
    if (errorFile.is_open()) {
        std::string line;
        while (std::getline(errorFile, line)) {
            stderr_result += line + "\n";
        }
        errorFile.close();
        std::remove(tempErrorFile.c_str());
    }

    return {stdout_result, stderr_result};
}

int main(int argc, char **argv) {
    argparse::ArgumentParser program("aps-opt", "0.1");

    program.add_argument("-i", "--input")
        .required()
        .help("Input CADL/MLIR file (e.g., examples/burst_demo.cadl or examples/zyy.mlir)");
    program.add_argument("-o", "--output")
        .required()
        .help("Output path (e.g., zyy_out)");
    program.add_argument("--clock")
        .required()
        .default_value(std::string("10.0"))
        .help("Clock period in ns (default: 10.0)");
    program.add_argument("--resource")
        .required()
        .default_value(std::string("examples/resource_ihp130.json"))
        .help("Path to resource.json for scheduling information");
    program.add_argument("--gensg")
        .default_value(false)
        .implicit_value(true)
        .help("Generate schedule graph and estimate the running cycle");
    program.add_argument("--genpr")
        .default_value(false)
        .implicit_value(true)
        .help("Generate pragma report");
    program.add_argument("--genrr")
        .default_value(false)
        .implicit_value(true)
        .help("Generate resource report");
    program.add_argument("--print-ir-after-all")
        .default_value(false)
        .implicit_value(true)
        .help("Print MLIR after each pass");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program << std::endl;
        return EXIT_FAILURE;
    }

    const auto inputFile = program.get<std::string>("--input");
    const auto clockPeriod = program.get<std::string>("--clock");
    const auto resourceFile = program.get<std::string>("--resource");
    const auto outputPath = program.get<std::string>("--output");
    const auto generateScheduleGraph = program.get<bool>("--gensg");
    const auto generatePragmaReport = program.get<bool>("--genpr");
    const auto printIrAfterAll = program.get<bool>("--print-ir-after-all");

    // Validate input file exists
    if (access(inputFile.c_str(), F_OK) != 0) {
        std::cerr << "[ERROR] Input file " << inputFile << " does not exist"
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Validate resource file exists
    if (access(resourceFile.c_str(), F_OK) != 0) {
        std::cerr << "[ERROR] Resource file " << resourceFile
                  << " does not exist" << std::endl;
        return EXIT_FAILURE;
    }

    // Check if input is a CADL file and convert to MLIR if necessary
    std::string mlirInput;

    // Check file extension
    if (inputFile.size() >= 5 && inputFile.substr(inputFile.size() - 5) == ".cadl") {
        std::cerr << "[INFO] Detected CADL input, converting to MLIR..." << std::endl;

        // Create output directory if it doesn't exist
        std::filesystem::create_directories(outputPath);

        // Convert input to absolute path
        std::string absolutePath = inputFile;
        if (inputFile[0] != '/') {
            char currentDir[PATH_MAX];
            if (getcwd(currentDir, sizeof(currentDir)) != nullptr) {
                absolutePath = std::string(currentDir) + "/" + inputFile;
            }
        }

        // Run pixi mlir command to convert CADL to MLIR (assumes pixi is in PATH)
        std::string convertCommand = "pixi run mlir " + absolutePath;
        auto [mlirOutput, mlirErrors] = executeCommand(convertCommand);

        // Check if conversion succeeded by looking at stdout
        // Note: pixi outputs task info to stderr, so we check stdout instead
        if (mlirOutput.empty()) {
            std::cerr << "[ERROR] CADL to MLIR conversion produced no output" << std::endl;
            if (!mlirErrors.empty()) {
                std::cerr << "Stderr output:" << std::endl;
                std::cerr << mlirErrors << std::endl;
            }
            return EXIT_FAILURE;
        }

        // If stderr contains error indicators (not just pixi task info), show them
        if (!mlirErrors.empty() &&
            (mlirErrors.find("ERROR") != std::string::npos ||
             mlirErrors.find("Error") != std::string::npos ||
             mlirErrors.find("error") != std::string::npos ||
             mlirErrors.find("Traceback") != std::string::npos)) {
            std::cerr << "[WARNING] Errors detected during CADL conversion:" << std::endl;
            std::cerr << mlirErrors << std::endl;
        }

        // Extract filename from input path
        std::string inputFilename = inputFile;
        size_t lastSlash = inputFile.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
            inputFilename = inputFile.substr(lastSlash + 1);
        }
        // Remove .cadl extension and add .mlir
        if (inputFilename.size() >= 5) {
            inputFilename = inputFilename.substr(0, inputFilename.size() - 5) + ".mlir";
        }

        // Write MLIR output to output folder
        std::string mlirFilePath = outputPath + "/" + inputFilename;
        std::ofstream mlirFile(mlirFilePath);
        if (!mlirFile.is_open()) {
            std::cerr << "[ERROR] Failed to create MLIR file: " << mlirFilePath << std::endl;
            return EXIT_FAILURE;
        }
        mlirFile << mlirOutput;
        mlirFile.close();

        mlirInput = mlirFilePath;
        std::cerr << "[INFO] CADL converted to MLIR and saved to: " << mlirFilePath << std::endl;
    } else {
        // Input is already MLIR
        mlirInput = inputFile;
    }

    // Build argument vector for hector_driver
    // Using std::vector of std::string for memory safety and RAII
    std::vector<std::string> args;

    args.push_back(argv[0]);
    args.push_back(mlirInput);
    args.push_back("--allow-unregistered-dialect");
    if (printIrAfterAll) {
        args.push_back("--mlir-print-ir-after-all");
    }
    args.push_back("--convert-math-to-call=resource=" + resourceFile);
    args.push_back("--demangle-func-name");
    args.push_back("--aps-hoist-readrf");
    args.push_back("--memory-map");
    args.push_back("--scf-for-index-cast");
    args.push_back("--aps-mem-to-memref");
    args.push_back("--canonicalize");
    args.push_back("--raise-scf-to-affine");
    args.push_back("--canonicalize");
    args.push_back("--normalize-memref-indices");
    args.push_back("--canonicalize");
    args.push_back("--affine-raise-from-memref");
    args.push_back("--infer-affine-mem-access");
    args.push_back("--canonicalize");
    args.push_back("--hls-unroll");
    args.push_back("--cse");
    args.push_back("--canonicalize");
    args.push_back("--affine-loop-normalize");
    args.push_back("--canonicalize");
    args.push_back("--new-array-partition");
    args.push_back("--canonicalize");
    args.push_back("--affine-mem-to-aps-mem");
    args.push_back("--memref-to-aps-mem");
    args.push_back("--canonicalize");

    if (generateScheduleGraph) {
        args.push_back("--loop-tripcount");
    }

    // Lower affine to SCF
    args.push_back("--lower-affine-for");
    args.push_back("--canonicalize");
    args.push_back("--expression-balance");

    // Convert input with top function, clock, and resource parameters
    args.push_back("--convert-input=clock=" + clockPeriod +
                               " resource=" + resourceFile +
                               " output-path=" + outputPath + "/file");

    // Dump SCF to JSON
    args.push_back("--dump-scf=json=" + outputPath + "/scf.json");

    args.push_back("--canonicalize");
    args.push_back("--scf-to-tor");
    args.push_back("--canonicalize");
    args.push_back("--schedule-tor");
    args.push_back("--aps-split-memory-ops");
    args.push_back("--tor-time-graph");
    args.push_back("--canonicalize");

    if (generatePragmaReport) {
        args.push_back("--generate-pragma-report");
    }

    if (generateScheduleGraph) {
        // Generate timing graph output
        args.push_back("--count-cycles=output-dir=" + outputPath);
    }

    // Convert std::vector<std::string> to char** for C-style API
    std::vector<char *> hectorArgv;
    hectorArgv.reserve(args.size());
    for (auto &arg : args) {
        hectorArgv.push_back(const_cast<char *>(arg.c_str()));
    }

    // Call hector_driver with the constructed arguments
    aps_opt_driver(static_cast<int>(hectorArgv.size()), hectorArgv.data());

    return EXIT_SUCCESS;
}
