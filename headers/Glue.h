#ifndef GLUE_H
#define GLUE_H

#include "Model.h"

/* High-level demo + test helpers */

/* Create a synthetic binary classification dataset (linearly separable) in arena.
   Returns X (N x D) and y (N x 1). */
void glue_create_synthetic_data(Arena *arena, int N, int D, Tensor *X_out, Tensor *y_out);

#endif // GLUE_H