#include "../../headers/core/Model.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

Model* modelCreate(Arena *arena, int *dims, int count) {
    Model *m = (Model*)arenaAlloc(arena, sizeof(Model));
    if (!m) return NULL;
    
    m->count = count - 1;
    m->layers = (Layer*)arenaAlloc(arena, sizeof(Layer) * m->count);
    if (!m->layers) return NULL;
    
    for (int i = 0; i < m->count; i++) {
        m->layers[i].w = tensorAlloc(arena, dims[i+1], dims[i]);
        m->layers[i].b = tensorAlloc(arena, dims[i+1], 1);
        m->layers[i].gradW = tensorAlloc(arena, dims[i+1], dims[i]);
        m->layers[i].gradB = tensorAlloc(arena, dims[i+1], 1);
        if (!m->layers[i].w || !m->layers[i].b) return NULL; 
        if (!m->layers[i].gradW || !m->layers[i].gradB) return NULL;  
        
        tensorFillXavier(m->layers[i].w, dims[i]); 

        for(int j = 0; j < m->layers[i].b->rows; j++) {
            m->layers[i].b->data[j] = 0.0f;
            m->layers[i].gradB->data[j] = 0.0f; 
        }
        for(int j = 0; j < m->layers[i].gradW->rows * m->layers[i].gradW->cols; j++) 
            m->layers[i].gradW->data[j] = 0.0f;
    }
    return m;
}

int modelSave(Model *m, const char *filename) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) { fprintf(stderr, "Error: Could not open %s for writing\n", filename); return -1; }
    FILE *f = fdopen(fd, "wb");
    if (!f) { fprintf(stderr, "Error: Could not open %s for writing\n", filename); close(fd); return -1; }
    
    fwrite(&m->count, sizeof(int), 1, f);
    
    // save dimensions for each layer (for verification on load)
    for (int i = 0; i < m->count; i++) {
        fwrite(&m->layers[i].w->rows, sizeof(int), 1, f);
        fwrite(&m->layers[i].w->cols, sizeof(int), 1, f);
    }
    
    for (int i = 0; i < m->count; i++) {
        // save weights
        fwrite(m->layers[i].w->data, sizeof(float), (size_t)m->layers[i].w->rows * (size_t)m->layers[i].w->cols, f);
        // save biases
        fwrite(m->layers[i].b->data, sizeof(float), m->layers[i].b->rows, f);
    }
    fclose(f);
    printf("Model saved to %s\n", filename);
    return 0;
}

// kind of lazy with cleanups... should be improved.

int modelLoad(Model *m, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Error: Could not open %s\n", filename); return -1; }
    
    int count;
    if (fread(&count, sizeof(int), 1, f) != 1) { fprintf(stderr, "Error: Failed to read model count from %s\n", filename); fclose(f); return -1; }
    if (count != m->count) { fprintf(stderr, "Error: Model layer count mismatch (file: %d, expected: %d)\n", count, m->count); fclose(f); return -1; }

    // verify dimensions match!!! (or else, catastrophic failure...)
    for (int i = 0; i < m->count; i++) {
        int rows, cols;
        if (fread(&rows, sizeof(int), 1, f) != 1 || fread(&cols, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read layer %d dimensions from %s\n", i, filename);
            fclose(f);
            return -1;
        }
        if (rows != m->layers[i].w->rows || cols != m->layers[i].w->cols) {
            fprintf(stderr, "Error: Layer %d dimension mismatch! File: %dx%d, Expected: %dx%d\n", 
                    i, rows, cols, m->layers[i].w->rows, m->layers[i].w->cols);
            fprintf(stderr, "Hint: The saved model has different hidden size. Retrain with current config.\n");
            fclose(f);
            return -1;
        }
    }

    for (int i = 0; i < m->count; i++) {
        size_t w_size = (size_t)m->layers[i].w->rows * (size_t)m->layers[i].w->cols;
        size_t b_size = m->layers[i].b->rows;
        if (fread(m->layers[i].w->data, sizeof(float), w_size, f) != w_size) { fprintf(stderr, "Error: Failed to read weights for layer %d from %s\n", i, filename); fclose(f); return -1; }
        if (fread(m->layers[i].b->data, sizeof(float), b_size, f) != b_size) { fprintf(stderr, "Error: Failed to read biases for layer %d from %s\n", i, filename); fclose(f); return -1; }
    }
    fclose(f);
    printf("Model loaded from %s\n", filename);
    return 0;
}