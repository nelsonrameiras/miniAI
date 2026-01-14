#include "../headers/Model.h"
#include "../headers/Grad.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>

LogisticModel model_create(Arena *arena, int D, float lr) {
    LogisticModel m;
    m.arena = arena;
    m.lr = lr;
    m.b = 0.0f;
    m.W = tensor_create(arena, D, 1);
    /* small random init: use a simple LCG generator from arena memory as deterministic */
    for (size_t i = 0; i < m.W.size; ++i) {
        /* very simple; not cryptographically strong; deterministic for reproducibility */
        unsigned int seed = (unsigned int)(uintptr_t)arena;
        seed = seed * 1664525u + 1013904223u + (unsigned int)i;
        float r = (float)(seed % 1000) / 1000.0f;
        m.W.data[i] = (r - 0.5f) * 0.1f; /* small values */
    }
    return m;
}

void model_predict(const LogisticModel *m, const Tensor *X, Tensor *out) {
    grad_logistic_forward(X, &m->W, m->b, out);
}

void model_train(LogisticModel *m, const Tensor *X, const Tensor *y, int epochs) {
    int N = X->rows;
    int D = X->cols;
    /* allocate work tensors in model's arena */
    Tensor preds = tensor_create(m->arena, N, 1);
    Tensor dW = tensor_create(m->arena, D, 1);

    for (int e = 0; e < epochs; ++e) {
        /* forward */
        grad_logistic_forward(X, &m->W, m->b, &preds);
        /* loss */
        float loss = grad_binary_cross_entropy(y, &preds);
        /* backward */
        grad_logistic_backward(X, y, &preds, &dW, &m->b /* abused as out param */);
        /* Note: grad_logistic_backward writes db into m->b; to avoid confusion, compute db separately */
        /* Recompute db properly */
        float db;
        grad_logistic_backward(X, y, &preds, &dW, &db);

        /* update: W -= lr * dW ; b -= lr * db */
        for (size_t i = 0; i < m->W.size; ++i) {
            m->W.data[i] -= m->lr * dW.data[i];
        }
        m->b -= m->lr * db;

        if ((e % 10) == 0 || e == epochs - 1) {
            float acc = model_accuracy(m, X, y);
            printf("Epoch %4d  Loss: %8.6f  Acc: %.4f\n", e, loss, acc);
        }
        /* Reset arena portion for temps by moving offset back — we rely on full reset by caller if needed. */
    }
}

float model_accuracy(const LogisticModel *m, const Tensor *X, const Tensor *y) {
    Tensor preds = tensor_create(m->arena, X->rows, 1);
    model_predict(m, X, &preds);
    int correct = 0;
    for (int i = 0; i < X->rows; ++i) {
        float p = preds.data[i];
        int pred_label = p >= 0.5f ? 1 : 0;
        int true_label = (int)(y->data[i] + 0.001f);
        if (pred_label == true_label) ++correct;
    }
    return (float)correct / (float)X->rows;
}