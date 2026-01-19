#ifndef GRAD_H
#define GRAD_H
#include "Tensor.h"

void tensorReLUDerivative(Tensor *out, Tensor *z, Tensor *upstreamDelta);

#endif