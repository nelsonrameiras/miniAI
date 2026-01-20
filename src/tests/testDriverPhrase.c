#include "../AIHeader.h"
#include "../headers/Segmenter.h"

/*
 * testDriverPhrase.c
 * 
 * Reads a PNG image containing a phrase, segments it into individual characters,
 * runs each through the trained model, and outputs the recognized text.
 * 
 * Usage:
 *   ./testDriverPhrase <phrase.png> [options]
 * 
 * Options:
 *   digits      - Use digits-only model (8x8, 10 classes)
 *   alpha       - Use alphanumeric model (16x16, 62 classes) [default]
 *   --model <path> - Specify custom model file
 */

// Character maps (must match training)
static const char *ALPHA_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
static const char *DIGIT_MAP = "0123456789";

// Configuration for different modes
typedef struct {
    int inputSize;          // 64 for 8x8, 256 for 16x16
    int outputSize;         // 10 for digits, 62 for alphanumeric
    int gridSize;           // 8 or 16
    const char *modelFile;  // Path to trained model
    const char *charMap;    // Character mapping
    const char *modeName;   // Display name
} PhraseConfig;

static void printUsage(const char *progName) {
    printf("Usage: %s <phrase.png> [options]\n\n", progName);
    printf("Options:\n");
    printf("  digits         Use digits-only model (8x8, 10 classes)\n");
    printf("  alpha          Use alphanumeric model (16x16, 62 classes) [default]\n");
    printf("  --model <path> Specify custom model file\n");
    printf("\nExamples:\n");
    printf("  %s phrase.png\n", progName);
    printf("  %s numbers.png digits\n", progName);
    printf("  %s text.png --model my_model.bin\n", progName);
}

static PhraseConfig parseArgs(int argc, char **argv) {
    // Default: alphanumeric mode
    PhraseConfig cfg = {
        .inputSize = 256,
        .outputSize = 62,
        .gridSize = 16,
        .modelFile = "IO/models/alpha_brain_png.bin",
        .charMap = ALPHA_MAP,
        .modeName = "ALPHANUMERIC"
    };

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "digits") == 0) {
            cfg.inputSize = 64;
            cfg.outputSize = 10;
            cfg.gridSize = 8;
            cfg.modelFile = "IO/models/digit_brain_png.bin";
            cfg.charMap = DIGIT_MAP;
            cfg.modeName = "DIGITS";
        } else if (strcmp(argv[i], "alpha") == 0) {
            // Already default
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            cfg.modelFile = argv[++i];
        }
    }

    return cfg;
}

static void printRecognizedPhrase(CharSequence *seq, int *predictions, 
                                   float *confidences, const char *charMap) {
    printf("\n========================================\n");
    printf("         RECOGNIZED PHRASE\n");
    printf("========================================\n\n");

    // Build the phrase string
    char *phrase = (char*)malloc(seq->count + 1);
    if (!phrase) return;

    int phraseLen = 0;
    for (int i = 0; i < seq->count; i++) {
        if (seq->chars[i] == NULL) {
            // Space
            phrase[phraseLen++] = ' ';
        } else {
            phrase[phraseLen++] = charMap[predictions[i]];
        }
    }
    phrase[phraseLen] = '\0';

    printf("  \"%s\"\n\n", phrase);

    // Print detailed per-character results
    printf("--- Character Details ---\n");
    printf("Pos | Char | Confidence\n");
    printf("----|------|------------\n");

    int charPos = 0;
    for (int i = 0; i < seq->count; i++) {
        if (seq->chars[i] == NULL) {
            printf("%3d | [SP] |     -\n", charPos++);
        } else {
            printf("%3d |   %c  |   %5.1f%%\n", 
                   charPos++, charMap[predictions[i]], confidences[i] * 100);
        }
    }

    // Calculate average confidence (excluding spaces)
    float totalConf = 0;
    int numChars = 0;
    for (int i = 0; i < seq->count; i++) {
        if (seq->chars[i] != NULL) {
            totalConf += confidences[i];
            numChars++;
        }
    }

    if (numChars > 0) {
        printf("\nAverage Confidence: %.1f%%\n", (totalConf / numChars) * 100);
    }

    printf("========================================\n");

    free(phrase);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    const char *imagePath = argv[1];
    PhraseConfig cfg = parseArgs(argc, argv);

    printf("=== Phrase Recognition ===\n");
    printf("Mode: %s\n", cfg.modeName);
    printf("Model: %s\n", cfg.modelFile);
    printf("Input: %s\n\n", imagePath);

    // Initialize arenas
    Arena *perm = arenaInit(16 * MB);
    Arena *scratch = arenaInit(4 * MB);
    if (!perm || !scratch) {
        fprintf(stderr, "Error: Failed to initialize memory arenas\n");
        return 1;
    }

    // Load the image
    printf("--- Loading Image ---\n");
    RawImage *img = imageLoad(imagePath);
    if (!img) {
        fprintf(stderr, "Error: Could not load image: %s\n", imagePath);
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    printf("Image size: %dx%d, %d channels\n", img->width, img->height, img->channels);

    // Segment the phrase
    printf("\n--- Segmenting Phrase ---\n");
    SegmenterConfig segCfg = defaultSegmenterConfig(cfg.gridSize);
    CharSequence *seq = segmentPhrase(img, segCfg);
    imageFree(img);

    if (!seq || seq->count == 0) {
        fprintf(stderr, "Error: Could not segment any characters from image\n");
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }

    printf("Found %d elements (characters + spaces)\n", seq->count);

    // Load optimized parameters (to get correct hidden size)
    FILE *paramFile = NULL;
    if (cfg.outputSize == 10) {
        paramFile = fopen("IO/confs/best_config_DIGITS_PNG.txt", "r");
    } else {
        paramFile = fopen("IO/confs/best_config_ALPHA_PNG.txt", "r");
    }
    
    int hiddenSize = DEFAULT_HIDDEN;
    if (paramFile) {
        float lr;
        if (fscanf(paramFile, "%d\n%f", &hiddenSize, &lr) == 2) {
            printf("Loaded config: hidden=%d\n", hiddenSize);
        }
        fclose(paramFile);
    } else {
        printf("Warning: Config file not found, using default hidden=%d\n", hiddenSize);
    }

    // Create and load model
    printf("\n--- Loading Model ---\n");
    int dims[] = {cfg.inputSize, hiddenSize, cfg.outputSize};
    Model *model = modelCreate(perm, dims, NUM_DIMS);
    if (!model) {
        fprintf(stderr, "Error: Failed to create model\n");
        freeCharSequence(seq);
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }

    if (modelLoad(model, cfg.modelFile) != 0) {
        fprintf(stderr, "Error: Could not load model from %s\n", cfg.modelFile);
        fprintf(stderr, "Make sure you have trained the model first using testDriverPNG\n");
        freeCharSequence(seq);
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }

    // Recognize each character
    printf("\n--- Recognizing Characters ---\n");
    int *predictions = (int*)malloc(seq->count * sizeof(int));
    float *confidences = (float*)malloc(seq->count * sizeof(float));
    if (!predictions || !confidences) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        freeCharSequence(seq);
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }

    for (int i = 0; i < seq->count; i++) {
        if (seq->chars[i] == NULL) {
            // Space - no prediction needed
            predictions[i] = -1;
            confidences[i] = 0;
            continue;
        }

        arenaReset(scratch);

        // Create input tensor from character data
        Tensor *input = tensorAlloc(scratch, cfg.inputSize, 1);
        if (!input) {
            fprintf(stderr, "Error: Failed to allocate input tensor\n");
            continue;
        }

        memcpy(input->data, seq->chars[i], cfg.inputSize * sizeof(float));

        // Get prediction
        float confidence;
        int prediction = gluePredict(model, input, scratch, &confidence);

        predictions[i] = prediction;
        confidences[i] = confidence;

        // Debug: show each character being processed
        printf("  Char %d: '%c' (%.1f%% confidence)\n", 
               i, cfg.charMap[prediction], confidence * 100);
    }

    // Print final result
    printRecognizedPhrase(seq, predictions, confidences, cfg.charMap);

    // Cleanup
    free(predictions);
    free(confidences);
    freeCharSequence(seq);
    arenaFree(perm);
    arenaFree(scratch);

    return 0;
}