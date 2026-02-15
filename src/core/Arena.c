#include "../headers/core/Arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// I was kind of lazy with the failure cleanups... ought be improved in the future.

Arena* arenaInit(size_t capacity) {
    Arena *arena = (Arena*)malloc(sizeof(Arena));
    if (!arena) { fprintf(stderr, "Error: Failed to allocate Arena struct\n"); return NULL; }
    
    arena->capacity = capacity;
    arena->used = 0;

    arena->buffer = (uint8_t*)malloc(capacity);
    if (!arena->buffer) { fprintf(stderr, "Error: Failed to allocate Arena buffer of %zu bytes\n", capacity); free(arena); return NULL; }
    
    return arena;
}

void* arenaAlloc(Arena *arena, size_t size) {
    // Basic 8-byte alignment
    size = (size + 7) & ~7;
    if (arena->used + size > arena->capacity) { fprintf(stderr, "Error: Arena out of memory (requested %zu, available %zu)\n", size, arena->capacity - arena->used); return NULL; }
    
    void *ptr = arena->buffer + arena->used;
    arena->used += size;
    memset(ptr, 0, size);
    return ptr;
}

void arenaReset(Arena *arena) {
    arena->used = 0;
}

void arenaFree(Arena *arena) {
    if (!arena) return;
    free(arena->buffer);
    free(arena);
}

size_t arenaRemainingCapacity(Arena *arena) {
    return arena->capacity - arena->used;
}