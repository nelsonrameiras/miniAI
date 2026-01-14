#include "../headers/Tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

Tensor tensor_create(Arena *arena, int rows, int cols) {
    Tensor t;
    t.rows = rows;
    t.cols = cols;
    t.size = (size_t)rows * (size_t)cols;
    t.arena = arena;
    t.data = (float *)arena_alloc(arena, t.size * sizeof(float));
    if (!t.data) {
        fprintf(stderr, "tensor_create: arena out of memory\n");
        t.rows = t.cols = t.size = 0;
        t.data = NULL;
    }
    return t;
}

void tensor_zero(Tensor *t) {
    memset(t->data, 0, t->size * sizeof(float));
}

void tensor_fill(Tensor *t, float value) {
    for (size_t i = 0; i < t->size; ++i) t->data[i] = value;
}

void tensor_copy(const Tensor *src, Tensor *dst) {
    if (src->size != dst->size) {
        fprintf(stderr, "tensor_copy: size mismatch\n");
        return;
    }
    memcpy(dst->data, src->data, src->size * sizeof(float));
}

void tensor_add(const Tensor *a, const Tensor *b, Tensor *dst) {
    if (a->size != b->size || a->size != dst->size) {
        fprintf(stderr, "tensor_add: size mismatch\n");
        return;
    }
    for (size_t i = 0; i < a->size; ++i) dst->data[i] = a->data[i] + b->data[i];
}

void tensor_sub(const Tensor *a, const Tensor *b, Tensor *dst) {
    if (a->size != b->size || a->size != dst->size) {
        fprintf(stderr, "tensor_sub: size mismatch\n");
        return;
    }
    for (size_t i = 0; i < a->size; ++i) dst->data[i] = a->data[i] - b->data[i];
}

void tensor_mul_elem(const Tensor *a, const Tensor *b, Tensor *dst) {
    if (a->size != b->size || a->size != dst->size) {
        fprintf(stderr, "tensor_mul_elem: size mismatch\n");
        return;
    }
    for (size_t i = 0; i < a->size; ++i) dst->data[i] = a->data[i] * b->data[i];
}

void tensor_scale(const Tensor *a, float scalar, Tensor *dst) {
    if (a->size != dst->size) {
        fprintf(stderr, "tensor_scale: size mismatch\n");
        return;
    }
    for (size_t i = 0; i < a->size; ++i) dst->data[i] = a->data[i] * scalar;
}

void tensor_matmul(const Tensor *A, const Tensor *B, Tensor *C) {
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        fprintf(stderr, "tensor_matmul: shape mismatch (%d,%d) @ (%d,%d) -> (%d,%d)\n",
                A->rows, A->cols, B->rows, B->cols, C->rows, C->cols);
        return;
    }
    int M = A->rows;
    int K = A->cols;
    int N = B->cols;
    /* naive triple loop; clarity prioritized; small sizes expected */
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A->data[i * K + k] * B->data[k * N + j];
            }
            C->data[i * N + j] = sum;
        }
    }
}

void tensor_transpose(const Tensor *a, Tensor *dst) {
    if (dst->rows != a->cols || dst->cols != a->rows) {
        fprintf(stderr, "tensor_transpose: shape mismatch\n");
        return;
    }
    for (int i = 0; i < a->rows; ++i) {
        for (int j = 0; j < a->cols; ++j) {
            dst->data[j * dst->cols + i] = a->data[i * a->cols + j];
        }
    }
}

void tensor_sigmoid_inplace(Tensor *a) {
    for (size_t i = 0; i < a->size; ++i) {
        float x = a->data[i];
        a->data[i] = 1.0f / (1.0f + expf(-x));
    }
}

float tensor_sum_all(const Tensor *a) {
    float s = 0.0f;
    for (size_t i = 0; i < a->size; ++i) s += a->data[i];
    return s;
}

void tensor_print(const Tensor *a, const char *name) {
    printf("Tensor %s (%dx%d):\n", name, a->rows, a->cols);
    for (int i = 0; i < a->rows; ++i) {
        for (int j = 0; j < a->cols; ++j) {
            printf("%8.4f ", a->data[i * a->cols + j]);
        }
        printf("\n");
    }
}