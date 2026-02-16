#include "../../../headers/cli/Commands.h"
#include "../../../headers/dataset/Dataset.h"
#include "../../../headers/dataset/TestUtils.h"
#include "../../AIHeader.h"
#include "../../IO/MemoryDatasets.h"
#include <stdio.h>
#include <stdlib.h>

extern TrainingConfig g_trainConfig;

int cmdBenchmark(CommandArgs args) {
    printf("=== BENCHMARK MODE ===\n\n");
    
    // Create arenas
    Arena *perm = arenaInit(16 * MB);
    Arena *scratch = arenaInit(4 * MB);
    
    if (!perm || !scratch) {
        fprintf(stderr, "Error: Could not allocate memory\n");
        return 1;
    }
    
    // Determine dataset parameters
    const char *charMap;
    int outputSize;
    
    if (args.dataset == DATASET_SPEC_DIGITS) {
        charMap = CHARMAP_DIGITS;
        outputSize = 10;
    } else {
        charMap = CHARMAP_ALPHA;
        outputSize = 62;
    }
    
    // Create dataset
    Dataset *ds = NULL;
    
    if (args.useStatic) {
        // Static in-memory dataset
        printf("Dataset: Static in-memory\n");
        float *data = (args.dataset == DATASET_SPEC_DIGITS) ? (float*)digits : (float*)dataset;
        ds = datasetCreateMemory(args.gridSize * args.gridSize, outputSize, args.gridSize,
                                data, args.dataset == DATASET_SPEC_DIGITS ? "DIGITS" : "ALPHANUMERIC",
                                args.modelFile, charMap);
    } else if (args.dataPath) {
        // PNG directory
        printf("Dataset: PNG from %s\n", args.dataPath);
        ds = datasetCreatePNG(args.dataPath, args.gridSize, outputSize,
                             args.dataset == DATASET_SPEC_DIGITS ? "DIGITS" : "ALPHANUMERIC",
                             args.modelFile, charMap);
    } else {
        fprintf(stderr, "Error: No dataset specified (use --static or --data)\n");
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    if (!ds) {
        fprintf(stderr, "Error: Could not create dataset\n");
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    printf("Grid: %dx%d (%d inputs)\n", args.gridSize, args.gridSize, ds->inputSize);
    printf("Classes: %d\n", ds->outputSize);
    printf("Repetitions: %d\n\n", args.benchmarkReps);
    
    // Run benchmark
    runBenchmark(ds, perm, scratch, args.benchmarkReps);
    
    // Cleanup
    datasetFree(ds);
    arenaFree(perm);
    arenaFree(scratch);
    
    printf("\nBenchmark complete!\n");
    return 0;
}