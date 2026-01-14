#include "../headers/Glue.h"
#include "../headers/Tensor.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

/* Create synthetic data: We draw a random weight vector w_true and bias b_true, then label by sign(X @ w_true + b_true)
   For reproducibility we produce deterministic pseudorandom values derived from arena pointer. */
void glue_create_synthetic_data(Arena *arena, int N, int D, Tensor *X_out, Tensor *y_out) {
    Tensor X = tensor_create(arena, N, D);
    Tensor y = tensor_create(arena, N, 1);
    /* deterministic pseudo-random generator using linear congruential steps */
    unsigned int seed = (unsigned int)(uintptr_t)arena ^ 0xC0FFEE;
    float w_true[64];
    if (D > 64) D = 64; /* guard; although API expects reasonable D */
    for (int j = 0; j < D; ++j) {
        seed = seed * 1664525u + 1013904223u + (unsigned int)j;
        w_true[j] = ((float)(seed % 1000) / 1000.0f - 0.5f) * 2.0f; /* [-1,1] */
    }
    float b_true = 0.2f;

    for (int i = 0; i < N; ++i) {
        /* sample x */
        for (int j = 0; j < D; ++j) {
            seed = seed * 1664525u + 1013904223u + (unsigned int)(i + j);
            float xv = ((float)(seed % 1000) / 1000.0f - 0.5f) * 4.0f; /* wider range */
            X.data[i * D + j] = xv;
        }
        /* compute logit */
        float logit = 0.0f;
        for (int j = 0; j < D; ++j) logit += X.data[i * D + j] * w_true[j];
        logit += b_true;
        /* label with sigmoid probability > 0.5 */
        float p = 1.0f / (1.0f + expf(-logit));
        y.data[i] = p >= 0.5f ? 1.0f : 0.0f;
    }

    *X_out = X;
    *y_out = y;
}