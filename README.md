# miniAI

**miniAI — A Minimal Neural Network Infrastructure in Pure C**

`miniAI` is an academic project that implements a **fully connected feed-forward neural network with backpropagation**, written entirely in **ISO C** and deliberately avoiding external libraries or frameworks.

The core purpose of this project is **didactic and architectural**: to explore how modern machine-learning concepts (tensors, gradients, optimizers, memory arenas, training loops) can be implemented **explicitly and transparently** at a low level.

This is *not* a performance-optimized framework, but a **clarity-first infrastructure** designed to expose internal mechanics that are typically hidden behind high-level ML libraries.

---

## Table of Contents

- [Design Philosophy](#design-philosophy)
- [Overall Architecture](#overall-architecture)
- [Core Infrastructure Decisions](#core-infrastructure-decisions)
- [Neural Network Model](#neural-network-model)
- [Training and Evaluation Pipeline](#training-and-evaluation-pipeline)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Build and Execution](#build-and-execution)
- [Configuration and Hyperparameters](#configuration-and-hyperparameters)
- [Benchmarks](#benchmarks)
- [Educational Scope](#educational-scope)

---

## Design Philosophy

The guiding principles behind `miniAI` are:

1. **No hidden abstractions**  
    Every tensor allocation, gradient computation, and parameter update is explicit.

2. **Minimal dependencies**  
    Only the C standard library is used. No BLAS, no CUDA, no external ML tooling.

3. **Memory transparency**  
    Memory ownership and lifetime are always visible and deterministic.

4. **Infrastructure over features**  
    The focus is on *how* a neural network system is built, not on achieving state-of-the-art accuracy.

---

## Overall Architecture

At a high level, the system is composed of six core layers:

```
Dataset → Tensor Ops → Gradients → Model → Training Glue → Driver
```

Each layer is implemented as a separate module with a strict separation of concerns.

---

## Core Infrastructure Decisions

### 1. Manual Tensor System

Instead of using a general numerical library, `miniAI` defines its own `Tensor` abstraction.

Key design choices:
- Tensors are **flat contiguous buffers**
- Shape metadata is stored explicitly
- Operations are implemented procedurally (no operator overloading)
- Broadcasting is intentionally limited to preserve clarity

This allows:
- Predictable memory access
- Easy inspection during debugging
- A clear mapping between math and code

---

### 2. Explicit Gradient Handling (No Autograd Graph)

Gradients are not inferred dynamically via a computation graph. Instead:

- Each operation has a **forward** and **backward** implementation.
- Gradient buffers are pre-allocated.
- Backpropagation is performed layer by layer.

This approach:
- Avoids hidden state
- Mirrors how backpropagation is taught theoretically
- Makes gradient flow easy to reason about

---

### 3. Arena Allocator for Memory Management

Dynamic memory allocation (`malloc/free`) inside training loops is avoided.

Instead, the project uses a **linear arena allocator**:

- Large memory blocks are allocated once
- Objects are allocated sequentially
- Memory is released all at once by resetting the arena

Benefits:
- Deterministic memory lifetime  
- Zero fragmentation  
- Lower overhead during training and inference  
- Simplified reasoning about ownership  

This choice is particularly relevant in C, where memory safety is a core concern.

---

### 4. Clear Module Boundaries

Each major concern lives in its own module:

| Module         | Responsibility                |
|----------------|------------------------------|
| `Arena`        | Memory allocation            |
| `Tensor`       | Numerical storage and ops    |
| `Grad`         | Gradient propagation         |
| `Model`        | Network structure and parameters |
| `Glue`         | Training, testing, inference |
| `Utils`        | Randomness and helpers       |
| `TestDriver`   | CLI logic and orchestration  |

This modularity allows the system to be extended without rewriting core logic.

---

## Neural Network Model

### Topology

- Type: Fully connected feed-forward network
- Layers: `NUM_LAYERS = 3`

Typical layout:

```
Input → Hidden → Output
```

### Characteristics

- Dense linear layers
- Non-linear activation functions
- Categorical loss for classification
- Gradient descent optimization with learning-rate decay

Layer sizes and learning parameters are compile-time configurable.

---

## Training and Evaluation Pipeline

The training loop follows a classical structure:

1. Load dataset
2. Initialize model parameters
3. Forward pass
4. Loss computation
5. Backpropagation
6. Parameter update
7. Learning-rate decay
8. Periodic evaluation

Noise injection during training and testing is supported to evaluate robustness.

---

## Datasets

### Alphanumeric Dataset (Default)

- Input size: `8 × 8` (64 values)
- Output classes: `62`
  - `0–9`, `A–Z`, `a–z`
- Source: `IO/alphabet.h`
- Persisted model: `IO/alpha_brain.bin`

### Digits Dataset

- Input size: `5 × 5` (25 values)
- Output classes: `10`
- Persisted model: `IO/digit_brain.bin`

Datasets are embedded directly in header files to remove runtime dependencies.

---

## Project Structure

```
miniAI/
├── AIHeader.h              # Global configuration and hyperparameters
├── Makefile
├── testDriver.c            # Full training/testing driver
├── testDriverSimple.c      # Minimal example
│
├── headers/
│   ├── Arena.h
│   ├── Tensor.h
│   ├── Grad.h
│   ├── Model.h
│   ├── Glue.h
│   ├── Utils.h
│   └── TestDriver.h
│
├── src/
│   ├── Arena.c
│   ├── Tensor.c
│   ├── Grad.c
│   ├── Model.c
│   ├── Glue.c
│   └── Utils.c
│
├── IO/
│   ├── alphabet.h
│   ├── best_config_ALPHA.txt
│   ├── best_config_DIGITS.txt
│   └── *.bin
```

---

## Build and Execution

### Build


```bash
make
```

### Run (alphanumeric)

```bash
./bin/testDriver
```

### Digits dataset

```bash
./bin/testDriver digits
```

### Inference only

```bash
./bin/testDriver run
```

---

## Configuration and Hyperparameters

All core parameters live in `AIHeader.h`:

- Network dimensions
- Learning rate and decay schedule
- Number of epochs and passes
- Noise injection parameters
- Stress-test and benchmark settings

This makes experimentation reproducible and centralized.

---

## Benchmarks

Benchmark mode measures pure inference cost:

```bash
./bin/testDriver bench
./bin/testDriver bench digits
```

The benchmark isolates:

- Memory allocation strategy
- Tensor operations
- Forward-pass efficiency

---

## Educational Scope

This project is intentionally low-level and verbose.

It is well suited for:

- Students learning neural networks from first principles
- Systems programmers exploring ML internals
- Experimentation with alternative memory models
- Extension towards CNNs, RNNs, or custom optimizers