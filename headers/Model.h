#ifndef MODEL_H
#define MODEL_H
#include "Tensor.h"

typedef struct {
    Tensor *w; // Weights
    Tensor *b; // Bias
    Tensor *z; // Pre-activation cache
    Tensor *a; // Post-activation cache
} Layer;

typedef struct {
    Layer *layers;
    int count;
} Model;

Model* modelCreate(Arena *arena, int *dims, int count);

void modelSave(Model *m, const char *filename);
void modelLoad(Model *m, const char *filename);

#endif