#ifndef GRAD_H
#define GRAD_H
#include "Tensor.h"

float sigmoidDerivative(float x);
void tensorSigmoidPrime(Tensor *out, Tensor *in);
void tensorReLUDerivative(Tensor *out, Tensor *z, Tensor *upstreamDelta);

#endif