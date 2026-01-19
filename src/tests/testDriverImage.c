#include "../AIHeader.h"
#include "../headers/TestDriver.h"

// Character maps
const char *ALPHA_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const char *DIGIT_MAP = "0123456789";

// Global training configuration
TrainingConfig g_trainConfig;

// Structure for datasets loaded from PNG files
typedef struct {
    float **samples;     // Array of sample pointers
    int numSamples;      // Number of successfully loaded samples
    int totalSlots;      // Total allocated slots (for cleanup)
    int sampleSize;      // Size of each sample (64 or 25)
} PNGDataset;

// Prototypes
PNGDataset* loadPNGDataset(const char *dir, int gridSize, const char *map);
void freePNGDataset(PNGDataset *ds);
void trainModelFromPNG(Model *model, PNGDataset *ds, DatasetConfig cfg, Arena *scratch);
void testPerfectPNG(Model *model, PNGDataset *ds, DatasetConfig cfg, Arena *scratch);
void testRobustnessPNG(Model *model, PNGDataset *ds, DatasetConfig cfg, Arena *scratch);
void testWithExternalPNG(Model *model, const char *pngPath, DatasetConfig cfg, Arena *scratch);
void visualDemoPNG(Model *model, PNGDataset *ds, DatasetConfig cfg, Arena *scratch);

int main(int argc, char **argv) {
    srand(time(NULL));
    g_trainConfig = defaultTrainingConfig();
    Arena *perm    = arenaInit(16 * MB);  // 16MB for model
    Arena *scratch = arenaInit(4 * MB);   // 4MB scratch arena

    // Default configuration: alphanumeric
    int gridSize = 16;
    int inputSize = 256;
    int outputSize = 62;
    const char *name = "ALPHANUMERIC";
    const char *saveFile = "IO/models/alpha_brain_png.bin";
    const char *map = ALPHA_MAP;
    const char *pngDir = "IO/pngAlphaChars";

    const char *testPngPath = NULL;
    // Parse command-line arguments
    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "digits") == 0) {
            gridSize = 8;
            inputSize = 64;
            outputSize = 10;
            name = "DIGITS";
            saveFile = "IO/models/digit_brain_png.bin";
            map = DIGIT_MAP;
            pngDir = "IO/pngDigits";
        } else if(strcmp(argv[i], "--test-png") == 0 && i + 1 < argc) {
            testPngPath = argv[++i];
        }
    }

    printf("Dataset: %s\t", name);
    printf("Grid: %dx%d (%d inputs)\n", gridSize, gridSize, inputSize);
    printf("Classes: %d\t", outputSize);
    printf("PNG Directory: %s\n\n", pngDir);

    printf("--- LOADING PNG DATASET ---\n");
    PNGDataset *dataset = loadPNGDataset(pngDir, gridSize, map);
    printf("Loaded %d samples from PNG files\n\n", dataset->numSamples);

    // Dataset configuration
    DatasetConfig cfg = { .inputSize = inputSize, .outputSize = outputSize, .gridSide = gridSize,
        .data = NULL, /* Not used (PNG dataset) */ .name = name, .saveFile = saveFile, .map = map
    };

    loadBestParameters(cfg);

    // Create model
    int dims[] = { cfg.inputSize, g_trainConfig.hiddenSize, cfg.outputSize };
    Model *model = modelCreate(perm, dims, NUM_DIMS);

    int shouldTrain = 1;
    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "run") == 0) {
            shouldTrain = 0;
            printf("--- LOADING EXISTING MODEL ---\n");
            modelLoad(model, cfg.saveFile);
            printf("Model loaded from %s\n\n", cfg.saveFile);
            break;
        }
    }

    if (shouldTrain) trainModelFromPNG(model, dataset, cfg, scratch);

    // Evaluation tests
    testPerfectPNG(model, dataset, cfg, scratch);
    testRobustnessPNG(model, dataset, cfg, scratch);
    visualDemoPNG(model, dataset, cfg, scratch);

    // External PNG test (if provided)
    if (testPngPath) {
        printf("\n--- TESTING EXTERNAL PNG ---\n");
        testWithExternalPNG(model, testPngPath, cfg, scratch);
    }

    freePNGDataset(dataset);
    arenaFree(perm);
    arenaFree(scratch);

    return 0;
}

// ===== DATASET LOADING FUNCTIONS =====

PNGDataset* loadPNGDataset(const char *dir, int gridSize, const char *map) {
    PNGDataset *ds = (PNGDataset*)malloc(sizeof(PNGDataset));
    if (!ds) return NULL;

    int numChars = strlen(map);
    ds->samples = (float**)malloc(numChars * sizeof(float*));
    if (!ds->samples) { free(ds); return NULL; }
    ds->numSamples = 0;
    ds->totalSlots = numChars;  // Store for proper cleanup
    ds->sampleSize = gridSize * gridSize;

    PreprocessConfig cfg = { .targetSize = gridSize, .threshold = 0.5f, .invertColors = 0 };

    for (int i = 0; i < numChars; i++) {
        char filename[256];
        char c = map[i];

        if (gridSize == 16) snprintf(filename, sizeof(filename), "%s/%03d_%c.png", dir, c, c);
        else snprintf(filename, sizeof(filename), "%s/%c.png", dir, c);

        struct stat st;
        if (stat(filename, &st) != 0) { printf("  Warning: %s not found, skipping\n", filename); ds->samples[i] = NULL; continue; }

        RawImage *img = imageLoad(filename);
        if (!img) { printf("  Warning: Failed to load %s\n", filename); ds->samples[i] = NULL; continue; }

        float *processed = imagePreprocess(img, cfg);
        imageFree(img);

        if (!processed) { printf("  Warning: Failed to preprocess %s\n", filename); ds->samples[i] = NULL; continue; }

        ds->samples[i] = processed;
        ds->numSamples++;
    }

    return ds;
}

void freePNGDataset(PNGDataset *ds) {
    if (!ds) return;

    if (ds->samples) {
        // Iterate over all allocated slots, not just numSamples
        for (int i = 0; i < ds->totalSlots; i++) {
            if (ds->samples[i]) free(ds->samples[i]);
        }
        free(ds->samples);
    }

    free(ds);
}

// ===== TRAINING =====

void trainModelFromPNG(Model *model, PNGDataset *ds, DatasetConfig cfg, Arena *scratch) {
    printf("--- TRAINING PHASE (%s) ---\n", cfg.name);
    printf("Training from PNG files...\n\n");

    float lr = g_trainConfig.learningRate;
    int numChars = strlen(cfg.map);
    int indices[numChars];

    int validCount = 0;
    for(int i = 0; i < numChars; i++) 
        if (ds->samples[i] != NULL) indices[validCount++] = i;
    if (validCount == 0) { fprintf(stderr, "Error: No valid samples to train on!\n"); return; }

    printf("Training with %d valid samples\n\n", validCount);

    for (int pass = 0; pass < TOTAL_PASSES; pass++) {
        shuffle(indices, validCount);

        for (int i = 0; i < validCount; i++) {
            arenaReset(scratch);
            int idx = indices[i];
            glueTrainDigit(model, ds->samples[idx], idx, lr, TRAIN_NOISE, scratch);
        }

        if (pass > 0 && pass % (DECAY_STEP / validCount) == 0) {
            lr *= DECAY_RATE;

            arenaReset(scratch);
            int testIdx = indices[rand() % validCount];
            Tensor *input = tensorAlloc(scratch, cfg.inputSize, 1);
            memcpy(input->data, ds->samples[testIdx], cfg.inputSize * sizeof(float));

            Tensor *output = glueForward(model, input, scratch);
            float loss = glueComputeLoss(output, testIdx, scratch);

            printf("Pass %d | Loss: %.6f | LR: %.6f\n", pass, loss, lr);
        }
    }

    modelSave(model, cfg.saveFile);
    printf("\nModel saved to: %s\n", cfg.saveFile);
}

// ===== TESTS =====

void testPerfectPNG(Model *model, PNGDataset *ds, DatasetConfig cfg, Arena *scratch) {
    printf("\n--- TEST 1: PERFECT SAMPLES (PNG) ---\n");
    int correct = 0;
    int tested = 0;
    int numChars = strlen(cfg.map);
    
    for (int t = 0; t < numChars; t++) {
        if (!ds->samples[t]) continue;  // Skip missing samples
        
        arenaReset(scratch);
        Tensor *input = tensorAlloc(scratch, cfg.inputSize, 1);
        memcpy(input->data, ds->samples[t], cfg.inputSize * sizeof(float));

        // printDigit(input->data, cfg.gridSide);
        
        float conf;
        int guess = gluePredict(model, input, scratch, &conf);
        
        if (guess == t) correct++;
        tested++;
        
        printf("Real: %c | AI: %c (Confidence: %.2f%%)\n", cfg.map[t], cfg.map[guess], conf * 100);
    }
    
    printf("\nAccuracy: %d/%d (%.1f%%)\n", correct, tested, (float)correct / tested * 100);
}

void testRobustnessPNG(Model *model, PNGDataset *ds, DatasetConfig cfg, Arena *scratch) {
    printf("\n--- TEST 2: STRESS TEST (SALT & PEPPER) ---\n");
    printf("Samples: %d | Noise: %d pixels\n\n", STRESS_TRIALS, (cfg.gridSide == 16) ? 4 : 2);
    
    int correct = 0;
    int noiseLevel = (cfg.gridSide == 16) ? 4 : 2;
    int numChars = strlen(cfg.map);
    
    int validIndices[numChars];
    int validCount = 0;
    for (int i = 0; i < numChars; i++) 
        if (ds->samples[i] != NULL) validIndices[validCount++] = i;
    
    for (int i = 0; i < STRESS_TRIALS; i++) {
        arenaReset(scratch);
        
        int label = validIndices[rand() % validCount];
        
        Tensor *input = tensorAlloc(scratch, cfg.inputSize, 1);
        memcpy(input->data, ds->samples[label], cfg.inputSize * sizeof(float));
        
        for(int n = 0; n < noiseLevel; n++) {
            int pos = rand() % cfg.inputSize;
            input->data[pos] = 1.0f - input->data[pos];
        }
        
        if (gluePredict(model, input, scratch, NULL) == label) correct++;
    }
    
    printf("Robustness Score: %.2f%%\n", (float)correct / STRESS_TRIALS * 100);
}

void visualDemoPNG(Model *model, PNGDataset *ds, DatasetConfig cfg, Arena *scratch) {
    int numChars = strlen(cfg.map);
    
    int validIndices[numChars];
    int validCount = 0;
    for (int i = 0; i < numChars; i++) if (ds->samples[i] != NULL) validIndices[validCount++] = i;
    if (validCount == 0) return;
    
    int targetIdx = validIndices[rand() % validCount];
    
    printf("\n--- VISUAL DEMO: BROKEN '%c' ---\n", cfg.map[targetIdx]);
    arenaReset(scratch);
    
    Tensor *noisy = tensorAlloc(scratch, cfg.inputSize, 1);
    memcpy(noisy->data, ds->samples[targetIdx], cfg.inputSize * sizeof(float));
    
    noisy->data[0] = 1.0f - noisy->data[0];
    noisy->data[cfg.inputSize-1] = 1.0f - noisy->data[cfg.inputSize-1];
    noisy->data[cfg.inputSize/2] = 1.0f - noisy->data[cfg.inputSize/2];
    
    float conf;
    int guess = gluePredict(model, noisy, scratch, &conf);
    
    printDigit(noisy->data, cfg.gridSide);
    printf("AI Guess: %c (Confidence: %.2f%%)\n", cfg.map[guess], conf * 100);
}

void testWithExternalPNG(Model *model, const char *pngPath, DatasetConfig cfg, Arena *scratch) {
    printf("Loading: %s\n", pngPath);
    
    RawImage *img = imageLoad(pngPath);
    if (!img) { fprintf(stderr, "Error: Could not load %s\n", pngPath); return; }
    
    printf("Image: %dx%d, %d channels\n", img->width, img->height, img->channels);
    
    PreprocessConfig pcfg = { .targetSize = cfg.gridSide, .threshold = 0.5f, .invertColors = 0 };
    
    float *processed = imagePreprocess(img, pcfg);
    imageFree(img);
    
    if (!processed) { fprintf(stderr, "Error: Could not preprocess image\n"); return; }
    
    printf("\nProcessed image:\n");
    printDigit(processed, cfg.gridSide);
    
    arenaReset(scratch);
    Tensor *input = tensorAlloc(scratch, cfg.inputSize, 1);
    memcpy(input->data, processed, cfg.inputSize * sizeof(float));
    
    float confidence;
    int prediction = gluePredict(model, input, scratch, &confidence);
    
    printf("\n=== RESULT ===\n");
    printf("Character: %c\n", cfg.map[prediction]);
    printf("Confidence: %.2f%%\n", confidence * 100);
    
    free(processed);
}

void applyBestParameters(int bestH, float bestL, DatasetConfig cfg) {
    g_trainConfig.hiddenSize = bestH;
    g_trainConfig.learningRate = bestL;

    FILE *f = NULL;
    if (strcmp(cfg.name, "DIGITS") == 0) f = fopen("IO/confs/best_config_DIGITS_PNG.txt", "w");
    else if (strcmp(cfg.name, "ALPHANUMERIC") == 0) f = fopen("IO/confs/best_config_ALPHA_PNG.txt", "w");
    
    if (!f) { printf("Error: Could not open file to save optimized parameters for %s.\n", cfg.name); return; }

    if (f) {
        fprintf(f, "%d\n%f", bestH, bestL); fclose(f);
        printf("\nOptimized parameters saved in 'IO/confs/best_config_%s_PNG.txt'\n", strcmp(cfg.name, "DIGITS") == 0 ? "DIGITS" : "ALPHA");
    }
}

void loadBestParameters(DatasetConfig cfg) {
    FILE *f;
    if(strcmp(cfg.name, "DIGITS") == 0) f = fopen("IO/confs/best_config_DIGITS_PNG.txt", "r");
    else f = fopen("IO/confs/best_config_ALPHA_PNG.txt", "r");

    if (!f) { printf("Warning: Optimized parameter file not found for %s. Using defaults.\n", cfg.name); return; }

    if (f) {
        if (fscanf(f, "%d\n%f", &g_trainConfig.hiddenSize, &g_trainConfig.learningRate) == 2) 
            printf("Optimized parameters loaded: Hidden=%d, LR=%.3f\n", g_trainConfig.hiddenSize, g_trainConfig.learningRate);
        fclose(f);
    }
}