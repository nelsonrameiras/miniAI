#ifndef GRAD_H
#define GRAD_H

#include "Tensor.h"

/* Compute logistic predictions: sigmoid(X @ W + b) -> out
   X: (N x D), W: (D x 1), b: scalar as 1x1 tensor or float
   out: (N x 1)
*/
void grad_logistic_forward(const Tensor *X, const Tensor *W, float b, Tensor *out);

/* Compute gradients for logistic regression using binary cross-entropy.
   Inputs:
     X: (N x D)
     y: (N x 1) labels in {0,1}
     preds: (N x 1) predicted probabilities
   Outputs:
     dW: (D x 1)
     db: pointer to float
   Uses formulas:
     error = preds - y (N x 1)
     dW = X^T @ error / N
     db = sum(error) / N
*/
void grad_logistic_backward(const Tensor *X, const Tensor *y, const Tensor *preds, Tensor *dW, float *db_out);

/* Binary cross-entropy loss: average over N */
float grad_binary_cross_entropy(const Tensor *y, const Tensor *preds);

#endif // GRAD_H