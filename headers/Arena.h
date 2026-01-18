#ifndef ARENA_H
#define ARENA_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t capacity;
    size_t used;
    uint8_t *buffer;
} Arena;

Arena* arenaInit(size_t capacity);
void* arenaAlloc(Arena *arena, size_t size);
void arenaReset(Arena *arena);
void arenaFree(Arena *arena);

#endif