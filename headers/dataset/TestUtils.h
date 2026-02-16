#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "../core/Model.h"
#include "Dataset.h"
#include "../core/Arena.h"

// Training functions
void trainModel(Model *model, Dataset *ds, Arena *scratch);

// Testing functions
void testPerfect(Model *model, Dataset *ds, Arena *scratch);
void testRobustness(Model *model, Dataset *ds, Arena *scratch);
void displayConfusionMatrix(Model *model, Dataset *ds, Arena *scratch);
void visualDemo(Model *model, Dataset *ds, Arena *scratch);

// Benchmark functions
typedef struct {
    int hiddenSize;
    float learningRate;
    float avgScore;
    float stdDev;
} BenchmarkResult;

void runBenchmark(Dataset *ds, Arena *perm, Arena *scratch, int repetitions);
BenchmarkResult runSingleExperiment(int hiddenSize, float lr, Dataset *ds, 
                                   Arena *perm, Arena *scratch, int repetitions);

// Configuration management
void loadBestConfig(const char *configFile);
void saveBestConfig(const char *configFile, int hiddenSize, float lr);

#endif // TEST_UTILS_H