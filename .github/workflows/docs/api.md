---
layout: default
title: API Reference
permalink: /api.html
---

# API Reference

## Commands

### `train`

Train a new model (or retrain) on a dataset.

```bash
./miniAI train --dataset <digits|alpha> [--static | --data [path]] [--load]
```

| Flag | Description |
|------|-------------|
| `--dataset digits` | Train on digit recognition (0–9) |
| `--dataset alpha` | Train on alphanumeric recognition (0–9, A–Z, a–z) — **default** |
| `--static` | Use in-memory static dataset — **default when no mode given** |
| `--data [path]` | Use PNG dataset. Optional custom path; defaults to `IO/images/` |
| `--load` | Load existing model and continue training instead of starting fresh |

**Examples:**

```basha
# Static digits (fastest)
./miniAI train --dataset digits --static

# PNG alphanumeric (realistic)
./miniAI train --dataset alpha --data

# Custom PNG directory
./miniAI train --dataset digits --data /path/to/my/digits/

# Continue training an existing model
./miniAI train --dataset alpha --data --load
```

---

### `test`

Test a trained model on a full dataset or a single image.

```bash
# Test on dataset
./miniAI test [--model <path>] --dataset <type> [--static | --data [path]]

# Test on single image (PNG mode automatic)
./miniAI test [--model <path>] --image <path>
```

| Flag | Description |
|------|-------------|
| `--model <path>` | Path to `.bin` model file. Inferred from dataset type if omitted |
| `--dataset <type>` | Dataset type: `digits` or `alpha` |
| `--static` | Use static dataset |
| `--data [path]` | Use PNG dataset |
| `--image <path>` | Test on a single image (auto-selects PNG model) |

**Examples:**

```bash
# Test static digit model
./miniAI test --dataset digits --static

# Test PNG alpha model
./miniAI test --dataset alpha --data

# Predict a single image
./miniAI test --image my_char.png

# Explicit model path
./miniAI test --model IO/models/digit_brain_png.bin --image test.png
```

**Important:** Models are not interchangeable between static and PNG. Static models expect different input dimensions than PNG models.

---

### `benchmark`

Run a hyperparameter grid search to find the best hidden size and learning rate.

```bash
./miniAI benchmark --dataset <type> [--static | --data [path]] [--reps <n>]
```

| Flag | Description |
|------|-------------|
| `--reps <n>` | Number of repetitions per configuration (default: 3). More = more stable results |

Searches over:
- Hidden sizes: `{16, 32, 64, 128, 256, 512, 1024}`
- Learning rates: `{0.001, 0.005, 0.008, 0.01, 0.015, 0.02}`

The best configuration is automatically saved to `IO/configs/` and used by subsequent train/test runs.

**Examples:**

```bash
./miniAI benchmark --dataset digits --static --reps 3
./miniAI benchmark --dataset alpha --data --reps 5
```

---

### `recognize`

Recognize a full phrase (multiple characters) in a PNG image.

```bash
./miniAI recognize [--dataset <type>] [--model <path>] --image <path>
```

Automatically segments characters, runs each through the model, and assembles the result with space detection.

**Examples:**

```bash
# Recognize alphanumeric phrase (default)
./miniAI recognize --image IO/images/testPhrases/hello_world_train.png

# Recognize digits only
./miniAI recognize --dataset digits --image numbers.png
```

---

### `help`

Show usage information.

```bash
./miniAI help
```

---

## Global Options

| Flag | Description |
|------|-------------|
| `--dataset <type>` | Dataset type: `digits` or `alpha` (default: `alpha`) |
| `--data [path]` | Use PNG dataset. Path optional |
| `--static` | Use static in-memory dataset |
| `--model <path>` | Explicit path to model `.bin` file |
| `--image <path>` | Path to input PNG image |
| `--grid <size>` | Grid size: `5`, `8`, or `16` (default: auto-detected) |
| `--reps <n>` | Benchmark repetitions (default: `3`) |
| `--load` | Load existing model instead of creating a new one |
| `--verbose` | Verbose output |

---

## Model Files

| Model | Dataset | Input | Classes |
|-------|---------|-------|---------|
| `IO/models/digit_brain.bin` | Digits static | 5×5 = 25 | 10 |
| `IO/models/digit_brain_png.bin` | Digits PNG | 8×8 = 64 | 10 |
| `IO/models/alpha_brain.bin` | Alpha static | 8×8 = 64 | 62 |
| `IO/models/alpha_brain_png.bin` | Alpha PNG | 16×16 = 256 | 62 |

---

## Config Files

Best hyperparameters are saved to `IO/configs/` after benchmarking:

```
IO/configs/best_config_digits_static.txt
IO/configs/best_config_digits_png.txt
IO/configs/best_config_alpha_static.txt
IO/configs/best_config_alpha_png.txt
```

Format (2 lines):
```
256       ← hidden size
0.020000  ← learning rate
```

These are loaded automatically before training and testing. You can edit them manually.

---

## Docker

```bash
# Pull
docker pull nelsonramosua/miniai:latest

# Train (mount IO/ to persist models)
docker run --rm -v $(pwd)/IO:/app/IO nelsonramosua/miniai train --dataset digits --static

# Test
docker run --rm -v $(pwd)/IO:/app/IO nelsonramosua/miniai test --dataset digits --static

# Help
docker run --rm nelsonramosua/miniai help
```