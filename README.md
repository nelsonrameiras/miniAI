# miniAI

[![Language](https://img.shields.io/badge/language-C-blue.svg)](https://devdocs.io/c/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**miniAI** is an educational implementation of an artificial neural network built entirely from scratch in pure C, with no external machine learning library dependencies. The project demonstrates the fundamentals of deep learning through a complete feed-forward architecture, including forward propagation, backpropagation, L2 regularization, gradient clipping, and hyperparameter optimization.

## Key Features

### Neural Network Architecture
- **Multi-Layer Feed-Forward Network**: Flexible architecture with configurable number of hidden layers.
- **Activation Functions**: ReLU for hidden layers, Softmax for output layer.
- **Complete Backpropagation**: Backpropagation implementation with ReLU derivatives.
- **L2 Regularization**: Overfitting prevention through weight penalization.
- **Gradient Clipping**: Protection against exploding gradients.
- **Xavier/He Initialization**: Smart weight initialization for faster convergence.

### Recognition Capabilities
1. **Digit Recognition (0-9)**
   - 5×5 pixel grid (25 input features).
   - 10 output classes.
   - Training with data augmentation (salt & pepper noise).

2. **Alphanumeric Recognition (0-9, A-Z, a-z)**
   - 8×8 pixel grid (64 input features).
   - 62 output classes.
   - Support for uppercase and lowercase.

3. **Phrase Recognition**
   - Automatic character segmentation.
   - Word space detection.
   - PNG image processing.
   - Full phrase support.

### Memory Management
- **Arena Allocator**: Efficient and deterministic memory allocation system.
- **Scratch Arena**: Temporary recyclable memory between operations.
- **Permanent Arena**: Persistent memory for weights and model structures.

### Image Processing
- **PNG Loading**: Native PNG image support via stb_image.
- **Preprocessing**: Grayscale conversion, resizing and binarization.
- **Otsu Threshold**: Automatic adaptive binarization.
- **Character Segmentation**: Detection and extraction of individual characters in phrases.

### Evaluation Tools
- **Robustness Testing**: Evaluation with artificial noise (salt & pepper).
- **Confusion Matrix**: Classification error visualization.
- **Automatic Benchmarking**: Grid search for hyperparameter optimization.
- **Confidence Metrics**: Output probabilities for each prediction.

## Project Structure

```
miniAI/
├── headers/                    # Main module headers
│   ├── Arena.h                # Memory management (arena allocator)
│   ├── Tensor.h               # Matrix operations and activations
│   ├── Model.h                # Neural model structure
│   ├── Grad.h                 # Gradients and derivatives
│   ├── Glue.h                 # Training and inference API
│   ├── Utils.h                # Utility functions
│   ├── ImageLoader.h          # PNG image loading
│   ├── ImagePreprocess.h      # Image preprocessing
│   └── Segmenter.h            # Character segmentation in phrases
│
├── src/                       # Implementations
│   ├── Arena.c                # Arena allocator implementation
│   ├── Tensor.c               # Tensor operations
│   ├── Model.c                # Model creation, save/load
│   ├── Grad.c                 # Gradient calculation
│   ├── Glue.c                 # Forward/backward pass
│   ├── Utils.c                # Utility functions
│   └── tests/                 # Test drivers
│       ├── testDriverSimple.c # Basic digit training
│       ├── testDriver.c       # Complete demo with benchmarking
│       ├── testDriverImage.c  # PNG image recognition
│       └── testDriverPhrase.c # Phrase recognition
│
├── IO/                        # Input/Output and data
│   ├── ImageLoader.c          # PNG loader implementation
│   ├── ImagePreprocess.c      # Image preprocessing
│   ├── Segmenter.c            # Character segmentation
│   ├── alphabet.h             # 62-character dataset (8×8)
│   ├── pngDigits/             # PNG digit images (0-9)
│   ├── pngAlphaChars/         # Alphanumeric PNG images
│   ├── testPhrases/           # Phrase images for testing
│   ├── models/                # Saved trained models (.bin)
│   ├── confs/                 # Optimized configurations
│   └── external/              # External libraries
│       └── stb_image.h        # Header-only PNG loader
│
├── AIHeader.h                 # Main unified header
├── Makefile                   # Build system
└── README.md                  # This file
```

## Compilation

### Prerequisites
- C Compiler (GCC recommended, but you can change in Makefile).
- Make.
- POSIX system (Linux, macOS, WSL).
- Standard math library (`libm`).
- Python3 for the PNG generation PY scripts.

### Build

```bash
# Compile all executables
make

# Compile and run main demo
make run

# Run specific tests
make run-simple         # Basic digit test
make run-image          # PNG image test
make run-phrase         # Phrase recognition help

# Clean compiled files
make clean

# Clean trained models only
make clean-models

# Complete rebuild
make rebuild

# View complete help
make help
```

### Generated Executables

1. **`testDriverSimple`** - Basic digit training.
2. **`testDriver`** - Complete demo with benchmarking.
3. **`testDriverPNG`** - Individual PNG image recognition.
4. **`testDriverPhrase`** - Image phrase recognition (uses model generated by `testDriverPNG`).

## Usage

### 1. Basic Digit Training

This was the first «iteration» of the project (testDriverSimple): smaller input layer, smaller no. classes, etc.

```bash
# Train digit model from scratch
./testDriverSimple

# Load pre-trained model
./testDriverSimple run

# Run hyperparameter benchmark
./testDriverSimple bench
```

**Expected output:**
```
--- TRAINING PHASE ---
Pass 0 | Loss: 2.302585 | LR: 0.005000
Pass 500 | Loss: 0.123456 | LR: 0.003500
Pass 1000 | Loss: 0.045678 | LR: 0.002450
Pass 1500 | Loss: 0.012345 | LR: 0.001715

--- TEST 1: PERFECT DIGITS ---
Real: 0 | AI: 0 (Confidence: 99.87%)
Real: 1 | AI: 1 (Confidence: 99.92%)
...
Perfect Digit Accuracy: 10/10

--- TEST 2: STRESS TEST (SALT & PEPPER NOISE) ---
Robustness Score (2-pixel noise): 98.50%
```

### 2. PNG Image Recognition

```bash
# Test with specific image
make test-png PNG=IO/pngAlphaChars/065_A.png

# Or directly
./testDriverPNG IO/pngAlphaChars/065_A.png
```

**Output:**
```
--- PNG CHARACTER RECOGNITION ---
Loading image: IO/pngAlphaChars/065_A.png
Image size: 32x32 pixels, 1 channels
Preprocessing to 8x8 grid...

********
*     *
*     *
*******
*     *
*     *

AI Prediction: 'A' (Confidence: 99.23%)
```

### 3. Phrase Recognition

```bash
# Recognize phrase in image (alphanumeric mode - default)
make phrase IMG=IO/testPhrases/hello_world.png

# Recognize digits only
make phrase IMG=IO/testPhrases/numbers.png MODE=digits

# With custom model (most call binary directly):
# make sure the custom model is compatible...
./testDriverPhrase phrase.png --model custom_model.bin
```

**Output:**
```
========================================
         RECOGNIZED PHRASE
========================================

  "HELLO WORLD"

--- Character Details ---
Pos | Char | Confidence
----|------|------------
  0 |  H   |   98.45%
  1 |  E   |   99.12%
  2 |  L   |   97.89%
  3 |  L   |   98.23%
  4 |  O   |   99.56%
  5 |      |   (space)
  6 |  W   |   98.91%
  7 |  O   |   99.34%
  8 |  R   |   97.67%
  9 |  L   |   98.45%
 10 |  D   |   99.01%
```

### 4. Hyperparameter Benchmarking

The system includes a benchmarking tool that automatically tests different configurations to find the best hyperparameters (att: the only testDriver that doesn't have benchmarking is the one that tests phrase recognition, as that will use the model trained by testDriverImage / PNG; the other testDrivers all create different models (either in no. classes or in input size (and are saved with appropriate names))):

```bash
./testDriver bench
```

**Output:**
```
--- SCIENTIFIC AI BENCHMARK (N=3) ---
Hidden |  LR   |  Avg Score  |  Std Dev  | Status
-------|-------|-------------|-----------|--------
  16   | 0.001 |   85.23%    |   2.45    |
  16   | 0.005 |   92.67%    |   1.89    | <-- STABLE
  16   | 0.008 |   94.12%    |   3.21    | <-- UNSTABLE
  ...
 128   | 0.005 |   98.45%    |   1.23    | <-- STABLE
  ...

WINNER: Hidden=128, LR=0.005 (Avg: 98.45%).
Optimized parameters saved in 'best_config.txt'.
```

## Technical Architecture

### Tensor Structure

```c
typedef struct {
    int rows;        // Number of rows.
    int cols;        // Number of columns.
    float *data;     // Data in row-major! format.
} Tensor;
```

### Layer Structure

```c
typedef struct {
    Tensor *w;       // Weights.
    Tensor *b;       // Bias.
    Tensor *z;       // Pre-activation (cache).
    Tensor *a;       // Post-activation (cache).
} Layer;
```

### Model Structure

```c
typedef struct {
    Layer *layers;   // Array of layers.
    int count;       // Number of layers.
} Model;
```

### Forward Propagation

For each layer `i`:
1. **Linear**: `z[i] = W[i] * a[i-1] + b[i]`
2. **Activation**: 
   - Hidden layers: `a[i] = ReLU(z[i])`
   - Output layer: `a[i] = Softmax(z[i])`

```c
// Simplified example. Go to the actual function for further detail.
Tensor* glueForward(Model *m, Tensor *input, Arena *scratch) {
    Tensor *currentInput = input;
    for (int i = 0; i < m->count; i++) {
        // z = W × input + b
        tensorDot(m->layers[i].z, m->layers[i].w, currentInput);
        tensorAdd(m->layers[i].z, m->layers[i].z, m->layers[i].b);
        
        // Apply activation
        if (i < m->count - 1) {
            tensorReLU(m->layers[i].a, m->layers[i].z);  // Hidden layers
        } else {
            // Output layer (softmax applied externally)
            for(int j = 0; j < m->layers[i].z->rows; j++) 
                m->layers[i].a->data[j] = m->layers[i].z->data[j];
        }
        currentInput = m->layers[i].a;
    }
    return currentInput;
}
```

### Backpropagation

The implemented backpropagation algorithm includes:

1. **Loss Function**: Cross-Entropy with Softmax
   ```
   L = -log(p_correct)
   ```

2. **Output Gradient** (Softmax + Cross-Entropy):
   ```
   delta_output = p - target
   ```
   where `target` is one-hot encoded!

3. **Weight Update** with L2 regularization:
   ```
   grad(W) = delta * a_prev + lambda × W
   W <- W - lr * clip(grad(W))
   ```

4. **Bias Update**:
   ```
   b <- b - lr * delta
   ```

5. **Error Propagation** (ReLU derivative):
   ```
   delta_prev = (W^T * delta_current) .* ReLU'(z_prev)
   ```
   where `.*` is element-wise multiplication

### Activation Functions

#### ReLU (Rectified Linear Unit)
```c
float relu(float x) {
    return x > 0 ? x : 0;
}

// Derivative
float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}
```

#### Sigmoid
(Currently, sigmoid is not being used, but is left in ```Tensor.c``` and ```Grad.c``` for potential future use.)
```c
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Derivative
float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}
```

#### Softmax
```c
void tensorSoftmax(Tensor *out, Tensor *in) {
    float max = in->data[0];
    for (int i = 1; i < in->rows; i++) {
        if (in->data[i] > max) max = in->data[i];
    }
    
    float sum = 0;
    for (int i = 0; i < in->rows; i++) {
        out->data[i] = expf(in->data[i] - max);  // Numerical stability
        sum += out->data[i];
    }
    
    for (int i = 0; i < in->rows; i++) {
        out->data[i] /= sum;
    }
}
```

## Hyperparameter Configuration

### Main Parameters (AIHeader.h)

The parameters can be tuned here.

```c
// Network Architecture
#define DEFAULT_HIDDEN  1024     // Hidden layer neurons
#define NUM_DIMS        3        // Number of dimensions [input, hidden, output]

// Training Parameters
#define DEFAULT_LR      0.005f   // Initial learning rate
#define LAMBDA          0.0001f  // L2 regularization factor
#define GRAD_CLIP       1.0f     // Gradient clipping threshold
#define TRAIN_NOISE     0.10f    // Pixel flip probability

// Training Configuration
#define TOTAL_PASSES    2000     // Number of epochs
#define DECAY_STEP      5000     // Steps for LR decay
#define DECAY_RATE      0.7f     // LR decay factor

// Tests
#define STRESS_TRIALS   1000     // Robustness tests
#define STRESS_NOISE    2        // Noisy pixels
#define CONFUSION_TESTS 500      // Confusion matrix tests
```

### Dynamic Configuration

The system supports **runtime configuration** through `TrainingConfig`:

```c
typedef struct {
    int   hiddenSize;       // Hidden layer size.
    float learningRate;     // Learning rate.
    int   benchmarkReps;    // Benchmark repetitions.
} TrainingConfig;

// Usage:
TrainingConfig config = {
    .hiddenSize = 512,
    .learningRate = 0.008f,
    .benchmarkReps = 5
};
```

## Implementation Details

### Arena Allocator

The system uses a custom arena allocator for efficient memory management. It's a simple version, but, for the purposes of this project, well enough. Could be extended in the future.

```c
typedef struct {
    size_t capacity;    // Total capacity.
    size_t used;        // Bytes used.
    uint8_t *buffer;    // Memory buffer.
} Arena;

// Creation
Arena *arena = arenaInit(8 * MB);

// Allocation (no individual free)
float *data = (float*)arenaAlloc(arena, sizeof(float) * size);

// Reset (frees everything at once)
arenaReset(arena);

// Destruction
arenaFree(arena);
```

**Advantages:**
- Extremely fast allocation (just pointer arithmetic).
- No metadata overhead per allocation.
- No fragmentation.
- Frees everything at once (ideal for batch operations).

### Model Serialization

Models can be saved and loaded in binary format (```Model.c```):

```c
// Save model
modelSave(model, "IO/models/digit_brain.bin");

// Load model
Model *model = modelCreate(arena, dims, NUM_DIMS);
modelLoad(model, "IO/models/digit_brain.bin");
```

**Binary file format:**
```
[int32] count            - Number of layers.
[int32] rows[0]          - Layer 0 dimensions.
[int32] cols[0]
...
[int32] rows[n-1]        - Layer n-1 dimensions.
[int32] cols[n-1]
[float32[]] weights[0]   - Layer 0 weights.
[float32[]] bias[0]      - Layer 0 bias.
...
[float32[]] weights[n-1] - Layer n-1 weights.
[float32[]] bias[n-1]    - Layer n-1 bias.
```

### Image Processing

#### Preprocessing Pipeline

1. **Loading**: PNG -> RawImage (thank you, stb_image).
2. **Conversion**: RGB -> Grayscale (luminance).
3. **Binarization**: Automatic Otsu threshold.
4. **Resizing**: Resize to target grid.
5. **Normalization**: [0, 255] → [0.0, 1.0].

```c
typedef struct {
    int targetSize;      // Grid size (8 for 8x8).
    float threshold;     // Binarization threshold.
    int invertColors;    // 1 = invert colors.
} PreprocessConfig;

PreprocessConfig cfg = {
    .targetSize = 8,
    .threshold = 0.5f,
    .invertColors = 0
};

float *processed = imagePreprocess(rawImage, cfg);
```

#### Phrase Segmentation

The system includes automatic character segmentation in phrases:

```c
typedef struct {
    float **chars;      // Character array.
    int count;          // Number of characters.
    int capacity;       // Allocated capacity.
    int charSize;       // Size of each character.
} CharSequence;

SegmenterConfig cfg = defaultSegmenterConfig(8);
CharSequence *seq = segmentPhrase(image, cfg);
```

**Segmentation Algorithm:**
1. Binarize image.
2. Horizontal projection to detect text line.
3. Vertical projection to detect character columns.
4. Bounding box extraction (with Otsu threshold).
5. Resizing to uniform grid.
6. Space detection (gaps larger than threshold).

## Educational Concepts

Why I actually created this project.

### Feed-Forward Neural Networks

A feed-forward network consists of:
- **Input layer**: Receives features (image pixels).
- **Hidden layers**: Intermediate processing.
- **Output layer**: Produces class probabilities.

### Gradient Descent

Training uses Stochastic Gradient Descent (SGD) with:
- **Shuffle**: Random order of examples in each epoch.
- **Mini-batch**: One example at a time (online learning).
- **Learning rate decay**: Gradual LR reduction for fine convergence.

### Regularization

Techniques to prevent overfitting:

1. **L2 Regularization (Weight Decay)**
   ```
   Loss_total = Loss_CE + lambda * ||W||²
   ```
   Penalizes large weights, favoring simpler models.

2. **Gradient Clipping**
   ```
   if |grad(W)| > threshold:
       grad(W) ← threshold * sign(grad(W))
   ```
   This prevents gradient explosion.

3. **Data Augmentation**
   - Random pixel flipping (salt & pepper noise).
   - Increases model robustness.

### Xavier/He Initialization

Smart initialization based on layer size (see [Xavier/He Initialization](http://proceedings.mlr.press/v9/glorot10a.html)):

```c
void tensorFillXavier(Tensor *t, int inSize) {
    float scale = sqrtf(2.0f / (float)inSize);
    for (int i = 0; i < t->rows * t->cols; i++)
        t->data[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
}
```

Maintains constant activation variance across layers, facilitating training.

## Performance and Results

### Typical Metrics

For digit recognition (5×5, 10 classes):
- **Perfect Accuracy**: 100% (clean digits).
- **Noisy Accuracy**: 97-99% (2 pixels with noise).
- **Training Time**: ~2-5 seconds (2000 epochs).
- **Model Size**: approx. 100KB.

For alphanumeric recognition (8×8, 62 classes):
- **Perfect Accuracy**: 98-100% (clean characters).
- **Phrase Recognition**: 95-98% (real phrases, **generated with IO/generatePhrases.py**).
- **Training Time**: ~10-30 seconds (5000 epochs).
- **Model Size**: approx. 500KB.

### Hyperparameter Optimization

Typical benchmark results are, for example:

| Hidden Size | Learning Rate | Accuracy | Std Dev | Status |
|-------------|--------------|----------|---------|--------|
| 64          | 0.005        | 94.2%    | 2.1     |        |
| 128         | 0.005        | 97.8%    | 1.3     | Stable |
| 256         | 0.008        | 98.5%    | 2.9     | Unstable |
| 512         | 0.005        | 98.9%    | 1.1     | **Best** |
| 1024        | 0.005        | 98.7%    | 1.4     | Stable |

## Debugging and Troubleshooting

### Common Issues

I ran into some of these in development, so I will leave some caveats here:

#### 1. Low Accuracy
- **Cause**: Inadequate learning rate.
- **Solution**: Run `./testDriver bench` to find optimal LR (it will be auto applied (saved into IO/configs/best_config_*.txt and read in training (if it exists))).

#### 2. Model Load Fails
```
Error: Layer 0 dimension mismatch! File: 128x25, Expected: 256x25
```
- **Cause**: Model saved with different configuration (different layer sizes, or you're trying to use the simpler digits model with alphanumeric testing (or vice-versa)).
- **Solution**: Delete `IO/models/*.bin` or adjust `DEFAULT_HIDDEN` according to what you're testing.

#### 3. Segmentation Fault
- **Cause**: Most likely, arena too small.
- **Solution**: Increase capacity in `arenaInit()` on your testDriver arena initialization.

#### 4. NaN in Loss
- **Cause**: Exploding gradients.
- **Solution**: Either reduce learning rate or increase `GRAD_CLIP`. See what makes more sense in your case.

### Advanced Debugging

To do a more advanced debugging, add debug prints in forward pass, with, for example:

```c
void debugForward(Model *m, Tensor *input) {
    for (int i = 0; i < m->count; i++) {
        printf("Layer %d:\n", i);
        printf("  Z min/max: %.4f / %.4f\n", 
               minValue(m->layers[i].z), 
               maxValue(m->layers[i].z));
        printf("  A min/max: %.4f / %.4f\n", 
               minValue(m->layers[i].a), 
               maxValue(m->layers[i].a));
    }
}
```

## Possible Extensions

### Future Features

Some features I would like to add in the future, to further extend the capabilities of this «mini AI»:

1. **Advanced Architectures**
   - Convolutional Neural Networks (CNN).
   - Dropout layers.
   - Batch Normalization.

2. **Optimizers**
   - Adam optimizer.
   - RMSprop.
   - Momentum.

3. **Data Augmentation**
   - Rotation.
   - Scaling.
   - Translation.
   - Elastic deformation.

4. **Visualization**
   - Loss curve plotting.
   - Filter visualization.
   - Feature t-SNE.

5. **Dataset Loading**
   - MNIST loader.
   - CIFAR-10 loader.
   - CSV/NPY format.

### How to Contribute

To add new features:

1. Maintain zero-dependency philosophy.
2. Use arena allocator for allocations.
3. Document functions with comments.
4. Add tests in `src/tests/`.
5. Update this README.

- see contribution pipeline with more detail below.

## References and Resources

### Fundamental Concepts
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Michael Nielsen.
- [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow, Bengio, Courville.
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/) (I did not use a CNN, but there were some pretty good ideas here, which I used!).

### Implementation Techniques
- [Backpropagation Algorithm](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) - Yann LeCun.
- [Xavier Initialization](http://proceedings.mlr.press/v9/glorot10a.html) - Glorot & Bengio.
- [Adam Optimizer](https://arxiv.org/abs/1412.6980) - Kingma & Ba (future implementation!).

### Regularization
- [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) - Srivastava et al (shoutout to Toronto Uni colleagues!).
- [Batch Normalization](https://arxiv.org/abs/1502.03167) - Ioffe & Szegedy.

## Technical Notes

### Known Limitations

1. **Scalability**: Current system designed for small datasets.
2. **GPU**: No GPU acceleration support (CPU only).
3. **Datasets**: Requires pre-segmented or preprocessed images. If we feed the AI non-segmented or non-preprocessed images, it will have a very hard time recognizing them.
4. **Precision**: Uses float32 (may be limiting for deeper models).

### Design Decisions

1. **Zero Dependencies**: Only C stdlib and libm for maximum portability.
2. **Arena Allocator**: Trade-off between flexibility and performance.
3. **Row-Major**: Matrices in row-major for cache locality.
4. **Hardcoded Activations**: ReLU/Softmax hardcoded for simplicity.

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a branch for your feature (`git checkout -b feature/newFeature`).
3. Commit your changes (`git commit -m 'Add some newFeature'`).
4. Push to the branch (`git push origin feature/newFeature`).
5. Open a Pull Request, which I will, then, analyze and approve.

### Guidelines

- Keep code clean and well documented.
- Follow existing code style (**important**).
- Add tests for new features.
- Update documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Nelson Ramos**
- GitHub: [@nelsonramosua](https://github.com/nelsonramosua).
- LinkedIn: [Nelson Ramos](https://www.linkedin.com/in/nelsonrocharamos/).

## Acknowledgments

- [stb_image](https://github.com/nothings/stb) by Sean Barrett - PNG loading.
- Deep learning community for educational resources.
- All (eventual) contributors and testers.

---

*For questions, suggestions or bugs, please open an [issue](https://github.com/nelsonramosua/miniAI/issues).*
