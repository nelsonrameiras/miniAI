#include "../headers/Grad.h"
#include "../headers/Tensor.h"
#include <stdio.h>
#include <math.h>

/* out must be allocated (N x 1) */
void grad_logistic_forward(const Tensor *X, const Tensor *W, float b, Tensor *out) {
    /* temp = X @ W -> (N x 1) */
    tensor_matmul(X, W, out);
    /* add bias */
    for (int i = 0; i < out->rows; ++i) out->data[i] += b;
    /* sigmoid in-place */
    tensor_sigmoid_inplace(out);
}

float grad_binary_cross_entropy(const Tensor *y, const Tensor *preds) {
    int N = y->rows;
    float eps = 1e-7f;
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        float yi = y->data[i];
        float pi = preds->data[i];
        /* clamp pi */
        if (pi < eps) pi = eps;
        if (pi > 1.0f - eps) pi = 1.0f - eps;
        sum += -(yi * logf(pi) + (1.0f - yi) * logf(1.0f - pi));
    }
    return sum / (float)N;
}

/* Compute dW (D x 1) and db. dW must be allocated */
void grad_logistic_backward(const Tensor *X, const Tensor *y, const Tensor *preds, Tensor *dW, float *db_out) {
    int N = X->rows; /* samples */
    int D = X->cols; /* features */
    /* error = preds - y (N x 1) allocate temp in same arena */
    Tensor error = tensor_create(X->arena, N, 1);
    for (int i = 0; i < N; ++i) error.data[i] = preds->data[i] - y->data[i];

    /* dW = X^T @ error / N -> shapes (D x N) @ (N x 1) => (D x 1)
       We'll create Xt (D x N) as a temp and reuse arena. */
    Tensor Xt = tensor_create(X->arena, D, N);
    /* transpose X into Xt */
    for (int i = 0; i < X->rows; ++i)
        for (int j = 0; j < X->cols; ++j)
            Xt.data[j * N + i] = X->data[i * X->cols + j];

    /* multiply Xt @ error -> dW_temp (D x 1) */
    tensor_matmul(&Xt, &error, dW);

    /* scale by 1/N */
    float invN = 1.0f / (float)N;
    for (size_t i = 0; i < dW->size; ++i) dW->data[i] *= invN;

    /* db = sum(error)/N */
    float s = 0.0f;
    for (int i = 0; i < N; ++i) s += error.data[i];
    *db_out = s * invN;
}