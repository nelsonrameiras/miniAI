# Changelog

All notable changes to miniAI are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `--seed <n>` flag for reproducible training and benchmarking. Passing the same seed produces identical weight initialisation and training order across runs, making experiment comparisons meaningful.
- `--resume` flag for `train` command. Loads an existing model and continues training from where it left off, instead of starting from scratch.
- `--verbose` flag now works. Prints loss every 100 passes during training (in addition to the existing decay-step prints), plus a summary of training hyperparameters at the start.
- `version` command and `--version` / `-V` flag. Prints `miniAI 0.1.0`. Also accessible via `make version`.
- `MINIAI_VERSION` constant in `AIHeader.h` - single source of truth for the version string, used by both the CLI and `printUsage`.
- Unit test suite (`src/tests/Tests.c`, `src/tests/TestConfig.c`). 7 suites, 120+ assertions covering Arena, Tensor, Grad, Shuffle, Model, Glue, and ImagePreprocess. Run with `make unit-tests`.
- `make version` shortcut.
- Unit tests run automatically in CI (both `ci.yml` and `nightly.yml`).

### Fixed
- **Critical:** Double `rand()` call in benchmark stress test caused random pixels to be read from one index and written to a different index, making the benchmark noise test inconsistent with `testRobustness`. Both now flip the same pixel.
- **Critical:** Learning rate decay in `runSingleExperiment` (benchmark) was `DECAY_STEP / outputSize` times faster than in `trainModel`. Both now decay at the same `DECAY_STEP` interval. Benchmark configs saved from previous versions may reflect this difference. That was legacy code not updated.
- `actions/first-interaction`: altered workflow to work correctly.
- `discussion category forms`: removed invalid `category:` top-level key from all templates (not a documented field; caused templates to not appear in the GitHub UI).
- `Segmenter.c`: `realloc` pattern for `bounds` and `gaps` arrays now uses temporary variables to prevent memory leaks when one allocation fails.
- `Recognize.c`: removed useless ternary `phrase[i == 0 ? 0 : i]` — always equivalent to `phrase[i]`.
- `Makefile` `.PHONY`: removed non-existent targets; added all new targets.

### Changed
- `trainModel` progress output now includes total passes (`Pass 500/3000`) for clearer feedback.
- README updated: project structure reflects `src/tests/`, `headers/tests/`, and `headers/utils/Random.h`; Command Reference includes `--seed`, `--resume`, `--verbose`, `version`; Quick Start Examples use correct make targets; `TrainingConfig` docs show all 5 fields; new Unit Tests section with suite breakdown table.

---

## [0.1.0] — Initial Release

### Added
- Feed-forward neural network in pure C with no ML library dependencies.
- Arena allocator for deterministic, zero-fragmentation memory management.
- Digit recognition: 5×5 static dataset and 8×8 PNG dataset.
- Alphanumeric recognition (62 classes): 8×8 static and 16×16 PNG datasets.
- Phrase recognition with automatic character segmentation via vertical projection.
- Otsu thresholding, bilinear resize, and center-of-mass centering in the image preprocessing pipeline.
- Mini-batch SGD with gradient clipping, L2 regularisation, and learning rate decay.
- Hyperparameter benchmark: grid search over hidden sizes and learning rates, saves best config automatically.
- Unified CLI: `train`, `test`, `benchmark`, `recognize`, `help`.
- `--load` flag to skip training and test an existing model.
- Salt-and-pepper robustness test and confusion matrix.
- OpenMP parallelisation for `tensorDot` and `tensorReLUDerivative`.
- Multi-platform CI/CD (Ubuntu + macOS), nightly builds, CodeQL, Trivy, Valgrind, cppcheck.
- GitHub Pages documentation site.
- Docker image published to Docker Hub and GitHub Container Registry on release.
- Dependabot for GitHub Actions and Docker updates.