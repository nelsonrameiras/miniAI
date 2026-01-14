#ifndef MODEL_H
#define MODEL_H

#include "Tensor.h"
#include "Arena.h"

/* Simple logistic regression model container */

typedef struct LogisticModel {
    Tensor W; /* D x 1 */
    float b;  /* scalar */
    float lr; /* learning rate */
    Arena *arena; /* parameters allocated in this arena */
} LogisticModel;

/* Create a logistic model with D features and initial lr. Parameters are allocated in arena. */
LogisticModel model_create(Arena *arena, int D, float lr);

/* Train the model on (X,y) for `epochs` iterations (full-batch). Prints diagnostics to stdout. */
void model_train(LogisticModel *m, const Tensor *X, const Tensor *y, int epochs);

/* Predict probabilities into out (N x 1) */
void model_predict(const LogisticModel *m, const Tensor *X, Tensor *out);

/* Evaluate accuracy (0.5 threshold) */
float model_accuracy(const LogisticModel *m, const Tensor *X, const Tensor *y);

#endif // MODEL_H