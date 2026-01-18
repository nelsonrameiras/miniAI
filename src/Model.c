#include "../headers/Model.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

void tensorFillXavier(Tensor *t, int inSize) {
    float scale = sqrtf(2.0f / (float)inSize);
    for (int i = 0; i < t->rows * t->cols; i++) {
        t->data[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
    }
}

Model* modelCreate(Arena *arena, int *dims, int count) {
    Model *m = (Model*)arenaAlloc(arena, sizeof(Model));
    m->count = count - 1;
    m->layers = (Layer*)arenaAlloc(arena, sizeof(Layer) * m->count);
    for (int i = 0; i < m->count; i++) {
        m->layers[i].w = tensorAlloc(arena, dims[i+1], dims[i]);
        m->layers[i].b = tensorAlloc(arena, dims[i+1], 1);
        
        tensorFillXavier(m->layers[i].w, dims[i]); 
        for(int j=0; j < m->layers[i].b->rows; j++) m->layers[i].b->data[j] = 0.0f;
    }
    return m;
}

#include <stdio.h>

void modelSave(Model *m, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    fwrite(&m->count, sizeof(int), 1, f);
    for (int i = 0; i < m->count; i++) {
        // Save weights
        fwrite(m->layers[i].w->data, sizeof(float), m->layers[i].w->rows * m->layers[i].w->cols, f);
        // Save biases
        fwrite(m->layers[i].b->data, sizeof(float), m->layers[i].b->rows, f);
    }
    fclose(f);
    printf("Model saved to %s\n", filename);
}

void modelLoad(Model *m, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { printf("Error: Could not open %s\n", filename); return; }
    int count;
    if (fread(&count, sizeof(int), 1, f) != 1) { printf("Error: Failed to read model count from %s\n", filename); fclose(f); return; }
    if (count != m->count) { printf("Error: Model file architecture mismatch!\n"); fclose(f); return; }

    for (int i = 0; i < m->count; i++) {
        size_t w_size = m->layers[i].w->rows * m->layers[i].w->cols;
        size_t b_size = m->layers[i].b->rows;
        if (fread(m->layers[i].w->data, sizeof(float), w_size, f) != w_size) { printf("Error: Failed to read weights for layer %d from %s\n", i, filename); fclose(f); return; }
        if (fread(m->layers[i].b->data, sizeof(float), b_size, f) != b_size) { printf("Error: Failed to read biases for layer %d from %s\n", i, filename); fclose(f); return; }
    }
    fclose(f);
    printf("Model loaded from %s\n", filename);
}