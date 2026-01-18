#ifndef TD_H
#define TD_H

#include "../AIHeader.h"

typedef struct {
    int inputSize;
    int outputSize;
    int gridSide;
    float *data;
    const char *name;
    const char *saveFile;
    const char *map;
} DatasetConfig;

void trainModel(Model *model, DatasetConfig cfg, Arena *scratch);
void testPerfect(Model *model, DatasetConfig cfg, Arena *scratch);
void testRobustness(Model *model, DatasetConfig cfg, Arena *scratch);
void displayConfusionMatrix(Model *model, DatasetConfig cfg, Arena *scratch);
void visualDemo(Model *model, DatasetConfig cfg, Arena *scratch);

void runBenchmarkSuite(Arena *perm, Arena *scratch, DatasetConfig cfg);
float runExperiment(int h, float lr, Arena *p, Arena *s, DatasetConfig cfg);
void applyBestParameters(int bestH, float bestL, DatasetConfig cfg);
void loadBestParameters(DatasetConfig cfg);

#endif // TD_H