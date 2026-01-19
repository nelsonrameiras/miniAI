#include "../headers/Model.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

Model* modelCreate(Arena *arena, int *dims, int count) {
    Model *m = (Model*)arenaAlloc(arena, sizeof(Model));
    if (!m) return NULL;
    
    m->count = count - 1;
    m->layers = (Layer*)arenaAlloc(arena, sizeof(Layer) * m->count);
    if (!m->layers) return NULL;
    
    for (int i = 0; i < m->count; i++) {
        m->layers[i].w = tensorAlloc(arena, dims[i+1], dims[i]);
        m->layers[i].b = tensorAlloc(arena, dims[i+1], 1);
        if (!m->layers[i].w || !m->layers[i].b) return NULL;
        
        tensorFillXavier(m->layers[i].w, dims[i]); 
        for(int j=0; j < m->layers[i].b->rows; j++) m->layers[i].b->data[j] = 0.0f;
    }
    return m;
}

int modelSave(Model *m, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Error: Could not open %s for writing\n", filename); return -1; }
    
    fwrite(&m->count, sizeof(int), 1, f);
    for (int i = 0; i < m->count; i++) {
        // Save weights
        fwrite(m->layers[i].w->data, sizeof(float), m->layers[i].w->rows * m->layers[i].w->cols, f);
        // Save biases
        fwrite(m->layers[i].b->data, sizeof(float), m->layers[i].b->rows, f);
    }
    fclose(f);
    printf("Model saved to %s\n", filename);
    return 0;
}

int modelLoad(Model *m, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Error: Could not open %s\n", filename); return -1; }
    int count;
    if (fread(&count, sizeof(int), 1, f) != 1) { fprintf(stderr, "Error: Failed to read model count from %s\n", filename); fclose(f); return -1; }
    if (count != m->count) { fprintf(stderr, "Error: Model file architecture mismatch!\n"); fclose(f); return -1; }

    for (int i = 0; i < m->count; i++) {
        size_t w_size = m->layers[i].w->rows * m->layers[i].w->cols;
        size_t b_size = m->layers[i].b->rows;
        if (fread(m->layers[i].w->data, sizeof(float), w_size, f) != w_size) { fprintf(stderr, "Error: Failed to read weights for layer %d from %s\n", i, filename); fclose(f); return -1; }
        if (fread(m->layers[i].b->data, sizeof(float), b_size, f) != b_size) { fprintf(stderr, "Error: Failed to read biases for layer %d from %s\n", i, filename); fclose(f); return -1; }
    }
    fclose(f);
    printf("Model loaded from %s\n", filename);
    return 0;
}