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

int modelSave(Model *m, const char *filename);  // Returns 0 on success, -1 on error
int modelLoad(Model *m, const char *filename);  // Returns 0 on success, -1 on error

#endif