---
layout: default
title: Architecture
permalink: /architecture.html
---

# Architecture

For visual diagrams of every layer, see **[Diagrams](diagrams.html)** — system architecture, data flows, backpropagation, memory model, and module dependencies.

## System Overview

miniAI is organized into four layers that communicate top-down:

```
CLI Layer          -> Parses commands and arguments.
Dataset Layer      -> Loads and prepares training data.
Core Layer         -> Neural network: forward pass, backprop, gradients.
Image Layer        -> PNG loading, preprocessing, segmentation.
```

---

## Directory Structure

```
miniAI/
├── headers/               # Public headers, mirroring src/
│   ├── core/              # Arena, Tensor, Model, Grad, Glue
│   ├── cli/               # ArgParse, Commands
│   ├── dataset/           # Dataset, TestUtils
│   ├── image/             # ImageLoader, ImagePreprocess, Segmenter
│   └── Utils.h
├── src/                   # Implementations
│   ├── core/
│   ├── cli/
│   │   └── commands/      # Train.c, Test.c, Benchmark.c, Recognize.c
│   ├── dataset/
│   └── image/
├── IO/
│   ├── MemoryDatasets.c/h # Hardcoded static datasets
│   ├── images/            # PNG training data
│   ├── models/            # Trained .bin model files
│   └── configs/           # Saved hyperparameter configs
├── docs/                  # Documentation source (this site)
├── tools/                 # Python scripts for PNG generation
├── AIHeader.h             # Central include — all constants and types
└── miniAI.c               # Entry point
```

---

## Core Components

### Arena Allocator (`src/core/Arena.c`)

Custom memory allocator. Instead of `malloc`/`free` per object, all allocations come from a pre-allocated slab. Reset is O(1) — just resets a pointer.

Two arenas are used at runtime:
- **`perm`** — persistent (model weights, layer structures)
- **`scratch`** — temporary (activations per forward pass, reset between iterations)

```c
Arena *perm    = arenaInit(16 * MB);
Arena *scratch = arenaInit(4 * MB);

// ... training loop ...
arenaReset(scratch);  // free all temporaries instantly
```

### Tensor (`src/core/Tensor.c`)

2D matrix in row-major layout. Holds a `float*` data pointer into an arena — no ownership, no free.

```c
typedef struct {
    int rows, cols;
    float *data;   // row-major: data[i*cols + j]
} Tensor;
```

Key operations: `tensorDot` (matrix multiply), `tensorAdd`, `tensorReLU`, `tensorSoftmax`, `tensorFillXavier`.

Inner loops are parallelized with `#pragma omp parallel for`.

### Model (`src/core/Model.c`)

Array of `Layer` structs. Each layer holds weight/bias tensors and their gradient accumulators.

```c
typedef struct {
    Tensor *w, *b;       // weights and bias
    Tensor *z, *a;       // pre/post activation cache
    Tensor *gradW, *gradB;
} Layer;
```

Serialized to binary `.bin` files. Format: layer count -> dimensions -> weights -> biases.

### Grad + Glue (`src/core/Grad.c`, `Glue.c`)

- **`Glue.c`** — forward pass, loss calculation (cross-entropy + softmax), weight update with L2 regularization and gradient clipping.
- **`Grad.c`** — backpropagation: delta propagation through ReLU derivative, gradient accumulation.

---

## Data Flow

### Training

```
Dataset (static or PNG)
  ↓  shuffle samples each epoch
Forward pass  →  cross-entropy loss
  ↓
Backward pass  →  gradients
  ↓
Weight update  →  SGD + L2 + gradient clip
  ↓
Save model to IO/models/*.bin
Save config to IO/configs/*.txt
```

### Inference (test / recognize)

```
Image or dataset sample
  ↓  preprocess (grayscale → binarize → resize → normalize)
Forward pass
  ↓  softmax probabilities
Top-1 prediction + confidence
```

### Phrase Recognition

```
Input PNG
  ↓  binarize
  ↓  vertical projection → character bounding boxes
  ↓  gap detection → spaces
For each character segment:
  ↓  resize to grid
  ↓  forward pass
  ↓  top-1 prediction
Assemble string
```

---

## Hyperparameter Configuration (`AIHeader.h`)

All compile-time defaults live in `AIHeader.h`. Runtime overrides come from `IO/configs/*.txt`.

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_HIDDEN` | 512 | Hidden layer neurons |
| `DEFAULT_LR` | 0.02 | Initial learning rate |
| `LAMBDA` | 0.0001 | L2 regularization factor |
| `GRAD_CLIP` | 5.0 | Gradient clipping threshold |
| `TOTAL_PASSES` | 3000 | Training epochs |
| `DECAY_STEP` | 500 | LR decay interval |
| `DECAY_RATE` | 0.7 | LR decay multiplier |
| `BATCH_SIZE` | 32 | Mini-batch size |
| `TRAIN_NOISE` | 0.10 | Salt & pepper noise during training |

---

## Adding Something New

### New command

1. Create `src/cli/commands/MyCommand.c` + header.
2. Add dispatch case in `src/cli/Commands.c`.
3. Add flag parsing in `src/cli/ArgParse.c`.
4. Update help text.
5. Update `docs/api.md`.

### New activation function

1. Add to `src/core/Tensor.c` + `headers/core/Tensor.h`.
2. Wire up in forward pass (`Glue.c`) and backward pass (`Grad.c`).

### New dataset type

1. Add to `src/dataset/Dataset.c`.
2. Add static data in `IO/MemoryDatasets.c` (if static) or a new `IO/images/` subfolder (if PNG).
3. Add dataset enum and string in `headers/dataset/Dataset.h`.