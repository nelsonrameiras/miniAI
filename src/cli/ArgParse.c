#include "../../headers/cli/ArgParse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void printUsage(const char *progName) {
    printf("miniAI - Neural Network Training and Recognition Tool\n\n");
    printf("Usage: %s <command> [options]\n\n", progName);
    
    printf("Commands:\n");
    printf("  train      Train a new model\n");
    printf("  test       Test an existing model (on dataset or specific image)\n");
    printf("  benchmark  Run hyperparameter optimization\n");
    printf("  recognize  Recognize characters/phrases in images\n");
    printf("  help       Show this help message\n\n");
    
    printf("Options:\n");
    printf("  --dataset <type>    Dataset type: digits, alpha (default: alpha)\n");
    printf("  --data [path]       Use PNG dataset (optional path, defaults to standard dirs)\n");
    printf("  --static            Use static in-memory dataset\n");
    printf("  --model <path>      Path to model file\n");
    printf("  --image <path>      Path to image file (optional for test, required for recognize)\n");
    printf("  --grid <size>       Grid size: 8 or 16 (default: auto)\n");
    printf("  --reps <n>          Benchmark repetitions (default: 3)\n");
    printf("  --load              Load existing model instead of training\n");
    printf("  --verbose           Verbose output\n\n");
    
    printf("Dataset Types:\n");
    printf("  Static (--static):  In-memory arrays, faster training\n");
    printf("    - digits: 5x5 grid, model: digit_brain.bin\n");
    printf("    - alpha:  8x8 grid, model: alpha_brain.bin\n");
    printf("  PNG (--data):       From PNG files, realistic testing\n");
    printf("    - digits: 8x8 grid (IO/pngDigits), model: digit_brain_png.bin\n");
    printf("    - alpha:  16x16 grid (IO/pngAlphaChars), model: alpha_brain_png.bin\n");
    printf("  Note: Models trained on static datasets won't work with PNG and vice-versa!\n\n");
    printf("  Note: Testing with --image automatically uses PNG mode (use --static to override)\n\n");
    
    printf("Examples:\n");
    printf("  # Train with static dataset (5x5, fast)\n");
    printf("  %s train --dataset digits --static\n\n", progName);
    
    printf("  # Train with PNG dataset (8x8, uses IO/pngDigits)\n");
    printf("  %s train --dataset digits --data\n\n", progName);
    
    printf("  # Train with custom PNG directory\n");
    printf("  %s train --dataset digits --data custom/pngs\n\n", progName);
    
    printf("  # Test static model on static dataset\n");
    printf("  %s test --model models/digit_brain.bin --dataset digits --static\n\n", progName);
    
    printf("  # Test PNG model on PNG dataset (uses IO/pngDigits)\n");
    printf("  %s test --model models/digit_brain_png.bin --dataset digits --data\n\n", progName);
    
    printf("  # Test PNG model on image (PNG mode is automatic, passing --data not mandatory)\n");
    printf("  %s test --model models/digit_brain_png.bin --image test.png\n\n", progName);
    
    printf("  # Force static mode with image (rarely needed)\n");
    printf("  %s test --model models/digit_brain.bin --image test.png --static\n\n", progName);
    
    printf("  # Benchmark on PNG dataset\n");
    printf("  %s benchmark --dataset digits --data --reps 5\n\n", progName);
    
    printf("  # Recognize phrase (always uses PNG model)\n");
    printf("  %s recognize --model models/alpha_brain_png.bin --image phrase.png\n\n", progName);
}

void getDefaultPaths(DatasetSpec spec, int useStatic, const char **dataPath, 
                    const char **modelFile, int *gridSize) {
    switch(spec) {
        case DATASET_SPEC_DIGITS:
            if (useStatic) {
                // Static in-memory dataset (5x5)
                if (dataPath) *dataPath = NULL;  // Use in-memory
                if (modelFile) *modelFile = "IO/models/digit_brain.bin";
                if (gridSize) *gridSize = 5;
            } else {
                // PNG dataset (8x8)
                if (dataPath) *dataPath = "IO/images/digitsPNG";
                if (modelFile) *modelFile = "IO/models/digit_brain_png.bin";
                if (gridSize) *gridSize = 8;
            }
            break;
            
        case DATASET_SPEC_ALPHA:
            if (useStatic) {
                // Static in-memory dataset (8x8)
                if (dataPath) *dataPath = NULL;  // Use in-memory
                if (modelFile) *modelFile = "IO/models/alpha_brain.bin";
                if (gridSize) *gridSize = 8;
            } else {
                // PNG dataset (16x16)
                if (dataPath) *dataPath = "IO/images/alphanumericPNG";
                if (modelFile) *modelFile = "IO/models/alpha_brain_png.bin";
                if (gridSize) *gridSize = 16;
            }
            break;
            
        case DATASET_SPEC_CUSTOM:
        default:
            if (dataPath) *dataPath = NULL;
            if (modelFile) *modelFile = "IO/models/custom_brain.bin";
            if (gridSize) *gridSize = 8;
            break;
    }
}

CommandArgs parseArgs(int argc, char **argv) {
    CommandArgs args = {
        .command = CMD_INVALID,
        .dataset = DATASET_SPEC_ALPHA,
        .dataPath = NULL,
        .modelFile = NULL,
        .imageFile = NULL,
        .configFile = NULL,
        .gridSize = 0,
        .benchmarkReps = 3,
        .loadModel = 0,
        .useStatic = -1,  // -1 = not specified, will be determined later
        .verbose = 0
    };
    
    if (argc < 2) {
        args.command = CMD_HELP;
        return args;
    }
    
    // Parse command
    const char *cmd = argv[1];
    if (strcmp(cmd, "train") == 0) {
        args.command = CMD_TRAIN;
    } else if (strcmp(cmd, "test") == 0) {
        args.command = CMD_TEST;
    } else if (strcmp(cmd, "benchmark") == 0 || strcmp(cmd, "bench") == 0) {
        args.command = CMD_BENCHMARK;
    } else if (strcmp(cmd, "recognize") == 0 || strcmp(cmd, "rec") == 0) {
        args.command = CMD_RECOGNIZE;
    } else if (strcmp(cmd, "help") == 0 || strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        args.command = CMD_HELP;
        return args;
    } else {
        args.command = CMD_INVALID;
        return args;
    }
    
    // Parse options
    int i = 2;
    while (i < argc) {
        if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "digits") == 0) {
                args.dataset = DATASET_SPEC_DIGITS;
            } else if (strcmp(argv[i], "alpha") == 0) {
                args.dataset = DATASET_SPEC_ALPHA;
            } else args.dataset = DATASET_SPEC_CUSTOM;
            i++;
        } else if (strcmp(argv[i], "--data") == 0) {
            // Check if next argument is a path (not a flag)
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                // Custom path provided
                args.dataPath = argv[++i];
            } else {
                // No path provided, will use defaults
                args.dataPath = "";  // Empty string signals "use default"
            }
            if (args.useStatic == -1) args.useStatic = 0;  // --data implies PNG
            i++;
        } else if (strcmp(argv[i], "--static") == 0) {
            args.useStatic = 1; i++;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.modelFile = argv[++i]; i++;
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            args.imageFile = argv[++i]; i++;
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            args.configFile = argv[++i]; i++;
        } else if (strcmp(argv[i], "--grid") == 0 && i + 1 < argc) {
            args.gridSize = atoi(argv[++i]); i++;
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            args.benchmarkReps = atoi(argv[++i]); i++;
        } else if (strcmp(argv[i], "--load") == 0) {
            args.loadModel = 1; i++;
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            args.verbose = 1; i++;
        } else if (strcmp(argv[i], "digits") == 0) {
            // Legacy: support "command digits" format
            args.dataset = DATASET_SPEC_DIGITS; i++;
        } else if (strcmp(argv[i], "alpha") == 0) {
            // Legacy: support "command alpha" format
            args.dataset = DATASET_SPEC_ALPHA; i++;
        } else if (strcmp(argv[i], "run") == 0) {
            // Legacy: support "run" as load flag
            args.loadModel = 1; i++;
        } else {
            i++; // Unrecognized argument, skip it
        }
    }
    
    // Determine useStatic if not specified
    // Logic:
    // - If --image provided: default to PNG (realistic testing)
    // - If recognize command: always PNG (needs PNG models)
    // - If --data specified: PNG
    // - Otherwise: static (faster training/testing)
    if (args.useStatic == -1) {
        if (args.command == CMD_RECOGNIZE || args.dataPath != NULL || args.imageFile != NULL) {
            args.useStatic = 0;  // PNG
        } else {
            args.useStatic = 1;  // Static by default for train/test/benchmark without images
        }
    }
    
    // Apply defaults if not specified
    // Empty dataPath ("") means use defaults for PNG
    if (!args.modelFile || args.gridSize == 0 || 
        (args.dataPath != NULL && args.dataPath[0] == '\0')) {
        const char *defaultData, *defaultModel;
        int defaultGrid;
        getDefaultPaths(args.dataset, args.useStatic, &defaultData, &defaultModel, &defaultGrid);
        
        if (!args.modelFile) args.modelFile = defaultModel;
        if (args.gridSize == 0) args.gridSize = defaultGrid;
        
        // If dataPath is empty string, replace with default
        if (args.dataPath != NULL && args.dataPath[0] == '\0') {
            args.dataPath = defaultData;
        }
    }
    
    // Validation
    if (args.command == CMD_RECOGNIZE && !args.imageFile) {
        fprintf(stderr, "Error: recognize command requires --image option\n");
        args.command = CMD_INVALID;
    }
    
    return args;
}