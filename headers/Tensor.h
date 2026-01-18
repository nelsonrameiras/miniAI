#ifndef TENSOR_H
#define TENSOR_H
#include "Arena.h"

typedef struct {
    int rows;
    int cols;
    float *data;
} Tensor;

Tensor* tensorAlloc(Arena *arena, int rows, int cols);
void tensorFillRandom(Tensor *t);
void tensorDot(Tensor *out, Tensor *a, Tensor *b); // out = a * b
void tensorAdd(Tensor *out, Tensor *a, Tensor *b); // out = a + b
float sigmoid(float x);
void tensorSigmoid(Tensor *out, Tensor *in);
void tensorSoftmax(Tensor *out, Tensor *in);
void tensorReLU(Tensor *out, Tensor *in);

#endif