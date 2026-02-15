#include "../headers/core/Grad.h"
#include <math.h>

float sigmoidDerivative(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f - s);
}

void tensorSigmoidPrime(Tensor *out, Tensor *in) {
    for (int i = 0; i < in->rows * in->cols; i++) 
        out->data[i] = sigmoidDerivative(in->data[i]);
}

void tensorReLUDerivative(Tensor *out, Tensor *z, Tensor *upstreamDelta) {
    for (int i = 0; i < z->rows * z->cols; i++) 
        out->data[i] = (z->data[i] > 0.0f) ? upstreamDelta->data[i] : 0.0f;
}