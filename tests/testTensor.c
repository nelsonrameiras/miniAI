#include "headers/Arena.h"
#include "headers/Tensor.h"
#include <stdio.h>

int main() {
    Arena *a = arena_create(1024*16);
    Tensor A = tensor_create(a, 2, 3);
    Tensor B = tensor_create(a, 3, 1);
    /* fill A */
    for (int i = 0; i < 2*3; ++i) A.data[i] = (float)i + 1.0f;
    for (int i = 0; i < 3*1; ++i) B.data[i] = 1.0f;
    Tensor C = tensor_create(a, 2, 1);
    tensor_matmul(&A, &B, &C);
    tensor_print(&C, "C");
    arena_destroy(a);
    return 0;
}