#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>

typedef struct Arena Arena;

/* Create an arena of `size` bytes. Returns NULL on allocation failure. */
Arena *arena_create(size_t size);

/* Allocate `nbytes` from the arena. Returns NULL if out-of-space. */
void *arena_alloc(Arena *arena, size_t nbytes);

/* Reset the arena to reuse memory without freeing the backing buffer. */
void arena_reset(Arena *arena);

/* Destroy the arena, freeing the backing memory. */
void arena_destroy(Arena *arena);

/* Query remaining bytes in arena */
size_t arena_remaining(const Arena *arena);

#endif // ARENA_H