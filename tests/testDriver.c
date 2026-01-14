#include "../headers/Arena.h"
#include "../headers/Tensor.h"
#include "../headers/Model.h"
#include "../headers/Glue.h"
#include <stdio.h>
#include <time.h>

/* Simple tests and demo combined */
int main() {
    const size_t ARENA_SIZE = 1024 * 1024 * 1024; /* 4MB */
    Arena *arena = arena_create(ARENA_SIZE);
    if (!arena) { fprintf(stderr, "Failed to create arena\n"); return 1; }

    printf("=== mini_ai demo: Logistic Regression (synthetic) ===\n");
    int N = 512;
    int D = 8;

    Tensor X = tensor_create(arena, N, D);
    Tensor y = tensor_create(arena, N, 1);
    glue_create_synthetic_data(arena, N, D, &X, &y);

    /* create model */
    LogisticModel model = model_create(arena, D, 0.1f);

    clock_t t0 = clock();
    model_train(&model, &X, &y, 200);
    clock_t t1 = clock();

    double elapsed = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
    printf("Training elapsed: %.4fs\n", elapsed);

    float acc = model_accuracy(&model, &X, &y);
    printf("Final accuracy on training set: %.4f\n", acc);

    arena_destroy(arena);
    return 0;
}