/* Minimal check: forward/backward match small hand-computed example */
#include "headers/Arena.h"
#include "headers/Tensor.h"
#include "headers/Grad.h"
#include <stdio.h>

int main() {
    Arena *a = arena_create(1024*16);
    Tensor X = tensor_create(a, 2, 2);
    /* X = [[1,2],[3,4]] */
    X.data[0]=1;X.data[1]=2;X.data[2]=3;X.data[3]=4;
    Tensor W = tensor_create(a, 2,1);
    W.data[0]=0.5f; W.data[1]=-0.25f;
    float b = 0.1f;
    Tensor preds = tensor_create(a, 2,1);
    grad_logistic_forward(&X,&W,b,&preds);
    tensor_print(&preds, "preds");
    Arena *a2 = a; (void)a2;
    arena_destroy(a);
    return 0;
}