#include "../headers/Arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct Arena {
    unsigned char *buf;
    size_t capacity;
    size_t offset;
};

Arena *arena_create(size_t size) {
    Arena *a = (Arena *)malloc(sizeof(Arena));
    if (!a) return NULL;
    a->buf = (unsigned char *)malloc(size);
    if (!a->buf) { free(a); return NULL; }
    a->capacity = size;
    a->offset = 0;
    return a;
}

void *arena_alloc(Arena *arena, size_t nbytes) {
    if (arena->offset + nbytes > arena->capacity) return NULL;
    void *ptr = arena->buf + arena->offset;
    arena->offset += nbytes;
    /* align offset to 8 bytes for safety */
    size_t rem = arena->offset % 8;
    if (rem) arena->offset += (8 - rem);
    return ptr;
}

void arena_reset(Arena *arena) {
    arena->offset = 0;
}

size_t arena_remaining(const Arena *arena) {
    return arena->capacity - arena->offset;
}

void arena_destroy(Arena *arena) {
    if (!arena) return;
    free(arena->buf);
    free(arena);
}