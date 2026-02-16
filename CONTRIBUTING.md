# Contributing to miniAI

Thank you for your interest in contributing to miniAI! This document provides guidelines and best practices for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

### Our Standards

- Be respectful and inclusive.
- Focus on constructive feedback.
- Accept criticism gracefully.
- Prioritize the community's best interests.

### Unacceptable Behavior

- Harassment or discriminatory language.
- Personal attacks.
- Trolling or spam.
- Publishing others' private information.

## Getting Started

### Prerequisites

- C compiler (GCC or Clang).
- Make.
- Git.
- Basic understanding of neural networks (helpful but not required).

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/miniAI.git
cd miniAI

# Add upstream remote
git remote add upstream https://github.com/nelsonramosua/miniAI.git
```

### Build and Test

```bash
# Build
make clean && make

# Quick smoke test
./miniAI train --dataset digits --static
./miniAI test --dataset digits --static

# Verify all commands work
./miniAI help
```

## Development Workflow

### 1. Create a Branch

Always create a new branch for your work:

```bash
# Update your fork
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements
- `perf/` - Performance improvements

### 2. Make Changes

Follow the coding standards (see below) and commit regularly:

```bash
git add .
git commit -m "feat: add feature description"
```

**Commit message format:**
```
<type>: <short description>

<optional longer description>

<optional footer>
```

**Types:**
- `feat` - New feature.
- `fix` - Bug fix.
- `docs` - Documentation changes.
- `style` - Code style changes (formatting, etc.).
- `refactor` - Code refactoring.
- `test` - Adding/updating tests.
- `chore` - Maintenance tasks.
- `perf` - Performance improvements.

**Examples:**
```
feat: add Adam optimizer support

Implemented Adam optimizer as an alternative to SGD.
Includes adaptive learning rates and momentum.

Closes #42
```

```
fix: resolve memory leak in arena allocator

The arena wasn't properly freeing nested allocations.
Added proper cleanup in arenaFree().

Fixes #38
```

### 3. Keep Your Branch Updated

```bash
# Fetch latest changes
git fetch upstream

# Rebase your branch
git rebase upstream/main

# Resolve conflicts if any
```

### 4. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 5. Open a Pull Request

1. Go to the original repository on GitHub.
2. Click "New Pull Request".
3. Select your fork and branch.
4. Fill in the PR template.
5. Submit!

## Coding Standards

### Code Style

#### Indentation and Formatting
```c
// Use 4 spaces (no tabs)
void myFunction(int param) {
    if (condition) {
        doSomething();
    }
}

// Braces on same line for control structures
for (int i = 0; i < n; i++) {
    process(i);
}

// For simple nested conditional/iterative structures, you can condense into one line
for (int a = 0; a < n; a++) {
    for (int b = 0; b < n; a++) {
        if (a < 20) process(a); 
        for (c = b + 1; b < n) process(c);
    }
}

// You can also condense into a single line if it's just cleanup
char *abc = (char*)malloc(sizeof(char));
if (!abc) { fprintf(stderr, "Error: Failed to allocate memory for abc\n"); return NULL; }
    
```

#### Naming Conventions
```c
// Functions: camelCase
void tensorDot(Tensor *out, Tensor *a, Tensor *b);
float calculateLoss(Model *m, Tensor *target);

// Structs: PascalCase
typedef struct {
    int rows;
    int cols;
    float *data;
} Tensor;

// Constants: UPPER_SNAKE_CASE
#define DEFAULT_HIDDEN 512
#define MAX_ITERATIONS 1000

// Variables: camelCase
int hiddenSize = 128;
float learningRate = 0.01f;
```

#### Comments
```c
// Good: Explain WHY, not WHAT
// Use Xavier initialization to maintain gradient variance across layers
tensorFillXavier(layer->w, inputSize);

// Bad: Redundant comment
// Initialize weights
tensorFillXavier(layer->w, inputSize);

// Good: Complex logic explanation
// Clip gradients to prevent explosion in deep networks.
// Without clipping, large gradients can cause NaN values during training.
float clipped = fminf(fmaxf(grad, -GRAD_CLIP), GRAD_CLIP);
```

#### Function Documentation
```c
/**
 * Performs matrix multiplication: out = a × b
 * 
 * @param out Output tensor (rows = a->rows, cols = b->cols)
 * @param a First input tensor
 * @param b Second input tensor
 * 
 * Requirements:
 * - a->cols must equal b->rows
 * - out must be pre-allocated with correct dimensions
 * - Uses OpenMP for parallelization
 */
void tensorDot(Tensor *out, Tensor *a, Tensor *b);
```

### Memory Management

**Always use the arena allocator:**

```c
// CORRECT
Arena *arena = arenaInit(8 * MB);
float *data = (float*)arenaAlloc(arena, sizeof(float) * size);
// ... use data ...
arenaFree(arena);  // Free everything at once

// WRONG - Don't use malloc/free
float *data = malloc(sizeof(float) * size);
free(data);
```

**Arena best practices:**
```c
// Separate permanent and temporary memory
Arena *perm = arenaInit(16 * MB);    // For model weights
Arena *scratch = arenaInit(4 * MB);  // For activations

// Reset scratch between operations
for (int epoch = 0; epoch < epochs; epoch++) {
    arenaReset(scratch);  // Reuse memory
    // ... training step ...
}

// Free both when done
arenaFree(perm);
arenaFree(scratch);
```

### Error Handling

```c
// Check for errors and provide context
FILE *f = fopen(filename, "rb");
if (!f) {
    fprintf(stderr, "Error: Could not open file '%s'\n", filename);
    return -1;
}

// Validate inputs
if (a->cols != b->rows) {
    fprintf(stderr, "Error: Matrix dimensions incompatible: %dx%d × %dx%d\n",
            a->rows, a->cols, b->rows, b->cols);
    return -1;
}

// Free resources on error
Model *model = modelCreate(arena, dims, count);
if (!model) {
    fprintf(stderr, "Error: Could not create model\n");
    arenaFree(arena);  // Clean up
    return NULL;
}
```

### File Organization

When adding new files:

```
headers/
  category/
    YourNewHeader.h

src/
  category/
    YourNewFile.c
```

**Categories:**
- `core/` - Neural network core (tensors, models, gradients)
- `cli/` - Command-line interface
- `dataset/` - Dataset handling
- `image/` - Image processing
- `utils/` - Utility functions

## Testing Guidelines

### Testing Your Changes

#### 1. Compile Tests
```bash
make clean && make

# Should compile without warnings
# If you see warnings, fix them!
```

#### 2. Functional Tests
```bash
# Test all dataset types
./miniAI train --dataset digits --static
./miniAI train --dataset digits --data
./miniAI train --dataset alpha --static
./miniAI train --dataset alpha --data

# Test all commands
./miniAI test --dataset digits --static
./miniAI test --image IO/images/digitsPNG/5.png
./miniAI recognize --image IO/images/testPhrases/hello.png
./miniAI benchmark --dataset digits --static --reps 2
```

#### 3. Edge Cases
```bash
# Test error conditions
./miniAI train --dataset invalid       # Should show error
./miniAI test --model nonexistent.bin  # Should show error
./miniAI test --image nonexistent.png  # Should show error
```

#### 4. Memory Checks (Optional but highly recommended)
```bash
# On Linux with Valgrind installed, something like:
valgrind --leak-check=full ./miniAI train --dataset digits --static
```

### Adding Tests

When adding new features:

1. **Add test cases** in the appropriate command.
2. **Test edge cases** (invalid inputs, boundary conditions).
3. **Document test procedure** in your PR.

Example:
```c
// In Test.c or create new test file
void testNewFeature() {
    // Setup
    Arena *arena = arenaInit(1 * MB);
    
    // Test normal case
    Result *result = yourNewFeature(arena, validInput);
    assert(result != NULL);
    assert(result->value == expectedValue);
    
    // Test edge case
    result = yourNewFeature(arena, edgeCase);
    assert(result != NULL);
    
    // Test error case
    result = yourNewFeature(arena, invalidInput);
    assert(result == NULL);  // Should fail gracefully
    
    // Cleanup
    arenaFree(arena);
}
```

## Pull Request Process

### Before Submitting

Checklist:
- [ ] Code compiles without warnings.
- [ ] All tests pass.
- [ ] Code follows style guidelines.
- [ ] Comments added for complex logic.
- [ ] Documentation updated (README, help text, etc.).
- [ ] Commit messages are clear and descriptive.
- [ ] Branch is up-to-date with main.
- [ ] No unrelated changes included.

### PR Template

When you open a PR, you'll see a template. Please fill it out completely:

**Required sections:**
- Description of changes.
- Type of change (bug fix, feature, etc.).
- Testing performed.
- Related issues.

**Optional but appreciated:**
- Screenshots (for UI/output changes).
- Performance impact.
- Breaking changes (if any).

### Automated Checks

Your PR will automatically run CI/CD checks:

**Build** - Compiles on Linux and macOS.
**Tests** - Runs test suite.
**Code Quality** - Static analysis (cppcheck).
**Format Check** - Code style validation.

**All checks must pass before merge!**

If checks fail:
1. Review the error in the Actions tab.
2. Fix the issue locally.
3. Push the fix.
4. Checks run automatically again.

### Review Process

1. **Automated Checks** (5-10 minutes)
   - CI/CD runs automatically.
   - Results posted to PR.

2. **Code Review** (1-3 days)
   - Maintainer reviews code.
   - May request changes.
   - Discussion in PR comments.

3. **Revisions** (if needed)
   - Address feedback.
   - Push new commits.
   - Tag reviewer when ready.

4. **Approval & Merge**
   - Once approved &
   - All checks passing.
   - PR is merged!

## Issue Reporting

### Bug Reports

Use the [bug report template](https://github.com/nelsonramosua/miniAI/issues/new?template=bug_report.md):

**Include:**
- Clear bug description.
- Steps to reproduce.
- Expected vs actual behavior.
- Environment (OS, compiler version).
- Error messages (full output).

**Example:**
```
Title: Memory leak in phrase recognition

Description:
When running phrase recognition on multiple images,
memory usage increases and eventually crashes.

Steps to reproduce:
1. Run: for i in {1..100}; do ./miniAI recognize --image test.png; done
2. Monitor memory with htop
3. Observe increasing memory usage

Expected: Memory should stay constant
Actual: Memory increases by approx. 10MB per iteration

Environment:
- OS: Ubuntu 22.04
- Compiler: gcc 11.3.0
- miniAI version: main branch (commit abc123)

Error output:
[paste full error here]
```

### Feature Requests

Use the [feature request template](https://github.com/nelsonramosua/miniAI/issues/new?template=feature_request.md):

**Include:**
- Feature description
- Motivation (why it's needed)
- Proposed implementation
- Example usage

**Example:**
```
Title: Add Adam optimizer support

Description:
Add Adam optimizer as an alternative to SGD for training.

Motivation:
Adam often converges faster than SGD and requires less
hyperparameter tuning, making it more beginner-friendly.

Proposed implementation:
- Add AdamOptimizer struct in core/
- Maintain running averages of gradients
- Add --optimizer flag to train command

Example usage:
./miniAI train --dataset digits --optimizer adam
```

## Community

### Getting Help

- **Questions?** Open a [discussion](https://github.com/nelsonramosua/miniAI/discussions).
- **Bug?** Use [bug report](https://github.com/nelsonramosua/miniAI/issues/new?template=bug_report.md).
- **Feature idea?** Use [feature request](https://github.com/nelsonramosua/miniAI/issues/new?template=feature_request.md).

### Communication

- **GitHub Issues** - Bug reports and features.
- **GitHub Discussions** - Questions and ideas.
- **Pull Requests** - Code contributions.

### Recognition

Contributors are recognized through:
- GitHub contributor stats.
- Acknowledgments in README.
- Preserved git history.

## Additional Resources

### Learning Resources

- [C Programming](https://devdocs.io/c/)
- [Neural Networks Basics](http://neuralnetworksanddeeplearning.com/)
- [Git Workflow](https://guides.github.com/introduction/flow/)

### Project Resources

- [README](README.md) - Project overview
- [Architecture](README.md#technical-architecture) - Code structure
- [Examples](README.md#usage) - Usage examples

## Questions?

If anything is unclear:
1. Check existing issues/discussions.
2. Read the documentation.
3. Ask in discussions.
4. Open an issue if it's a bug.

Thank you for contributing to miniAI!

---

**Last updated**: 2026
**License**: [LICENSE](LICENSE)