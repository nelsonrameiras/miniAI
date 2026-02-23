#include "../../headers/core/Tensor.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

Tensor* tensorAlloc(Arena *arena, int rows, int cols) {
    Tensor *t = (Tensor*)arenaAlloc(arena, sizeof(Tensor));
    if (!t) return NULL;

    // Validate dimensions to prevent negative or zero sizes and integer overflow
    if (rows <= 0 || cols <= 0) { fprintf(stderr, "Error: tensorAlloc received non-positive dimensions (%d x %d)\n", rows, cols); return NULL; }
    // Compute total number of elements using size_t and check for overflow in nrows * ncols
    size_t nrows = (size_t)rows; size_t ncols = (size_t)cols;
    if (ncols != 0 && nrows > SIZE_MAX / ncols) { fprintf(stderr, "Error: tensorAlloc dimension overflow (%d x %d)\n", rows, cols); return NULL; }
    size_t elements = nrows * ncols;
    // Check for overflow when scaling by sizeof(float) 
    if (elements != 0 && elements > SIZE_MAX / sizeof(float)) { fprintf(stderr, "Error: tensorAlloc size overflow for (%d x %d)\n", rows, cols); return NULL; }

    t->rows = rows;
    t->cols = cols;
    t->data = (float*)arenaAlloc(arena, sizeof(float) * elements);
    if (!t->data) return NULL;

    return t;
}

void tensorDot(Tensor *out, Tensor *a, Tensor *b) {
    // validate dims: a->cols must equal b->rows for matrix multiplication
    if (a->cols != b->rows) { fprintf(stderr, "Error: tensorDot dimension mismatch: a(%d×%d) x b(%d×%d)\n", a->rows, a->cols, b->rows, b->cols); return; }
    
    #pragma omp parallel for schedule(static) // use openMP threading optimization
    for (int i = 0; i < a->rows; i++) {
        // inits row 'i' with zeros only once
        for (int j = 0; j < b->cols; j++) out->data[i * b->cols + j] = 0.0f;

        for (int k = 0; k < a->cols; k++) {
            float a_val = a->data[i * a->cols + k];
            for (int j = 0; j < b->cols; j++) 
                out->data[i * b->cols + j] += a_val * b->data[k * b->cols + j];
        }
    }
}

void tensorAdd(Tensor *out, Tensor *a, Tensor *b) {
    // Validate dimensions match
    if (a->rows != b->rows || a->cols != b->cols) { fprintf(stderr, "Error: tensorAdd dimension mismatch: a(%d×%d) + b(%d×%d)\n", a->rows, a->cols, b->rows, b->cols); return; }
    
    for (int i = 0; i < a->rows * a->cols; i++) 
        out->data[i] = a->data[i] + b->data[i];
}

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

void tensorSigmoid(Tensor *out, Tensor *in) {
    for (int i = 0; i < in->rows * in->cols; i++) 
        out->data[i] = sigmoid(in->data[i]);
}

void tensorSoftmax(Tensor *out, Tensor *in) {
    float maxVal = in->data[0];
    for (int i = 1; i < in->rows; i++) 
        if (in->data[i] > maxVal) maxVal = in->data[i];

    float sum = 0.0f;
    for (int i = 0; i < in->rows; i++) {
        out->data[i] = expf(in->data[i] - maxVal);
        sum += out->data[i];
    }

    for (int i = 0; i < in->rows; i++) 
        out->data[i] /= sum;
}

void tensorReLU(Tensor *out, Tensor *in) {
    for (int i = 0; i < in->rows * in->cols; i++) 
        out->data[i] = (in->data[i] > 0) ? in->data[i] : 0.0f;
}

void tensorFillXavier(Tensor *t, int inSize) {
    float scale = sqrtf(2.0f / (float)inSize);
    for (int i = 0; i < t->rows * t->cols; i++)
        t->data[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
}