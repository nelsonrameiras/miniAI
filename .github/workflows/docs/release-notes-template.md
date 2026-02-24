## What's New

See [CHANGELOG.md](https://github.com/nelsonramosua/miniAI/blob/main/CHANGELOG.md) for the full list of changes in this release.

## Features

- Complete neural network implementation in pure C.
- Digit recognition (5×5 static, 8×8 PNG).
- Alphanumeric recognition (8×8 static, 16×16 PNG).
- Phrase recognition with automatic segmentation.
- Hyperparameter optimization via benchmarking.
- Unified command-line interface.

## Downloads

Choose the binary for your platform:

- **Linux (x64)**: `miniAI-linux-x64.zip`
- **macOS (x64)**: `miniAI-macos-x64.zip`

Verify downloads with the included `.sha256` checksum files.

## Installation

```bash
# Extract
unzip miniAI-linux-x64.zip
cd miniAI

# Run
./miniAI help
```

## Quick Start

```bash
# Train a model
./miniAI train --dataset digits --static

# Test the model
./miniAI test --dataset digits --static

# Run benchmark
./miniAI benchmark --dataset digits --static --reps 3
```

## Documentation

- Included `README.md` — full project documentation.
- Online docs: https://nelsonramosua.github.io/miniAI/.

## Changelog