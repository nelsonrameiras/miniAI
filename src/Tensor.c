#include "../headers/Tensor.h"
#include <math.h>
#include <stdlib.h>

Tensor* tensorAlloc(Arena *arena, int rows, int cols) {
    Tensor *t = (Tensor*)arenaAlloc(arena, sizeof(Tensor));
    t->rows = rows;
    t->cols = cols;
    t->data = (float*)arenaAlloc(arena, sizeof(float) * rows * cols);
    return t;
}

void tensorFillRandom(Tensor *t) {
    for (int i = 0; i < t->rows * t->cols; i++) {
        t->data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

void tensorDot(Tensor *out, Tensor *a, Tensor *b) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            out->data[i * b->cols + j] = sum;
        }
    }
}

void tensorAdd(Tensor *out, Tensor *a, Tensor *b) {
    for (int i = 0; i < a->rows * a->cols; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

void tensorSigmoid(Tensor *out, Tensor *in) {
    for (int i = 0; i < in->rows * in->cols; i++) {
        out->data[i] = sigmoid(in->data[i]);
    }
}

void tensorSoftmax(Tensor *out, Tensor *in) {
    float maxVal = in->data[0];
    for (int i = 1; i < in->rows; i++) {
        if (in->data[i] > maxVal) maxVal = in->data[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < in->rows; i++) {
        out->data[i] = expf(in->data[i] - maxVal);
        sum += out->data[i];
    }

    for (int i = 0; i < in->rows; i++) {
        out->data[i] /= sum;
    }
}

void tensorReLU(Tensor *out, Tensor *in) {
    for (int i = 0; i < in->rows * in->cols; i++) {
        out->data[i] = (in->data[i] > 0) ? in->data[i] : 0.0f;
    }
}