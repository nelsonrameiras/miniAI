#ifndef GLUE_H
#define GLUE_H

#include "Model.h"
#include "Arena.h"

// High-level API for the test driver
Tensor* glueForward(Model *m, Tensor *input, Arena *scratch);

void glueAccumulateGradients(Model *m, float *rawData, int label, float noiseLevel, Arena *scratch);
void glueUpdateWeights(Model *m, float lr, int batchSize);

int gluePredict(Model *m, Tensor *input, Arena *scratch, float *outConfidence);

float glueComputeLoss(Tensor *output, int label, Arena *scratch);

#endif