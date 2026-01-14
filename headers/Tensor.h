#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include "Arena.h"

/* Simple 2D tensor abstraction (row-major). For this project we restrict to 2D tensors
   (vectors are Nx1). The API is explicit and minimal but adequate for logistic regression.
*/

typedef struct Tensor {
    float *data;      /* pointer into arena */
    int rows;
    int cols;
    size_t size;      /* rows * cols */
    Arena *arena;     /* owning arena */
} Tensor;

/* Create a tensor in arena with given shape. Data is uninitialized. */
Tensor tensor_create(Arena *arena, int rows, int cols);

/* Zero-initialize tensor */
void tensor_zero(Tensor *t);

/* Fill tensor with a constant */
void tensor_fill(Tensor *t, float value);

/* Copy src -> dst (dst must be allocated) */
void tensor_copy(const Tensor *src, Tensor *dst);

/* Basic ops: dst = a + b (elementwise) */
void tensor_add(const Tensor *a, const Tensor *b, Tensor *dst);

/* dst = a - b */
void tensor_sub(const Tensor *a, const Tensor *b, Tensor *dst);

/* dst = a * b (elementwise) */
void tensor_mul_elem(const Tensor *a, const Tensor *b, Tensor *dst);

/* dst = a * scalar */
void tensor_scale(const Tensor *a, float scalar, Tensor *dst);

/* dst = a @ b (matrix multiplication). Shapes must match. */
void tensor_matmul(const Tensor *a, const Tensor *b, Tensor *dst);

/* dst = transpose(a) (allocates dst) */
void tensor_transpose(const Tensor *a, Tensor *dst);

/* Apply sigmoid in-place */
void tensor_sigmoid_inplace(Tensor *a);

/* Sum along rows or all elements */
float tensor_sum_all(const Tensor *a);

/* Print (for debugging) */
void tensor_print(const Tensor *a, const char *name);

#endif // TENSOR_H