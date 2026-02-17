---
layout: default
title: Quick Start
permalink: /quickstart.html
---

# Quick Start Guide

## Installation

### Option 1: Download Release Binary

Download the latest pre-built binary for your platform:

- [Linux x64](https://github.com/nelsonramosua/miniAI/releases/latest)
- [macOS x64](https://github.com/nelsonramosua/miniAI/releases/latest)

```bash
unzip miniAI-linux-x64.zip
cd miniAI
./miniAI help
```

### Option 2: Build from Source

**Prerequisites:** GCC (or Clang on macOS), Make, Python 3 (optional, for PNG generation)

```bash
git clone https://github.com/nelsonramosua/miniAI.git
cd miniAI
make
./miniAI help
```

**macOS note:** OpenMP is handled automatically by the Makefile. If you hit issues, install `libomp`:
```bash
brew install libomp
```

### Option 3: Docker

```bash
docker pull nelsonramosua/miniai:latest
docker run --rm nelsonramosua/miniai help
```

---

## Your First Model

### Train on digits (fast, ~5 seconds)

```bash
./miniAI train --dataset digits --static
```

### Test the model

```bash
./miniAI test --dataset digits --static
```

### That's it — you trained a neural network in C!

---

## Common Workflows

### Static workflow (fast, for development)

```bash
# Train
./miniAI train --dataset digits --static

# Test
./miniAI test --dataset digits --static

# Find best hyperparameters
./miniAI benchmark --dataset digits --static --reps 3

# Test again with optimized config
./miniAI test --dataset digits --static
```

### PNG workflow (realistic images)

```bash
# Train
./miniAI train --dataset digits --data

# Test on dataset
./miniAI test --dataset digits --data

# Test on a single image
./miniAI test --image my_digit.png

# Benchmark
./miniAI benchmark --dataset digits --data --reps 3
```

### Alphanumeric + phrase recognition

```bash
# Train alphanumeric PNG model
./miniAI train --dataset alpha --data

# Recognize a phrase in an image
./miniAI recognize --image IO/images/testPhrases/hello_world_train.png
```

---