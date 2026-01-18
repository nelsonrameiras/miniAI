#include "../headers/Arena.h"
#include <stdlib.h>
#include <string.h>

Arena* arenaInit(size_t capacity) {
    Arena *arena = (Arena*)malloc(sizeof(Arena));
    arena->capacity = capacity;
    arena->used = 0;
    arena->buffer = (uint8_t*)malloc(capacity);
    return arena;
}

void* arenaAlloc(Arena *arena, size_t size) {
    // Basic 8-byte alignment
    size = (size + 7) & ~7;
    if (arena->used + size > arena->capacity) return NULL;
    void *ptr = arena->buffer + arena->used;
    arena->used += size;
    memset(ptr, 0, size);
    return ptr;
}

void arenaReset(Arena *arena) {
    arena->used = 0;
}

void arenaFree(Arena *arena) {
    free(arena->buffer);
    free(arena);
}