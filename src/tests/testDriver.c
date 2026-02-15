#include "../AIHeader.h"
#include "../headers/TestDriver.h"
#include "../IO/alphabet.h" 
#include "../IO/digits.h"

static const char *ALPHA_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
static const char *DIGIT_MAP = "0123456789";

// Global training configuration
TrainingConfig g_trainConfig;

int main(int argc, char **argv) {
    srand(time(NULL));
    g_trainConfig = defaultTrainingConfig();

    DatasetConfig cfg = {64, 62, 8, (float*)dataset, "ALPHANUMERIC", "IO/models/alpha_brain.bin", ALPHA_MAP};

    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "digits") == 0) {
            cfg.inputSize = 25; cfg.outputSize = 10; cfg.gridSide = 5;
            cfg.data = (float*)digits; cfg.name = "DIGITS";
            cfg.saveFile = "IO/models/digit_brain.bin"; cfg.map = DIGIT_MAP;
            break;
        }
    }
    loadBestParameters(cfg);

    Arena *perm = arenaInit(8 * 1024 * 1024);
    Arena *scratch = arenaInit(2 * 1024 * 1024);

    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        if ((argv[2] && strcmp(argv[2], "digits") != 0) || argv[3]) 
            g_trainConfig.benchmarkReps = argv[3] ? atoi(argv[3]) : atoi(argv[2]);
        if (g_trainConfig.benchmarkReps <= 0) g_trainConfig.benchmarkReps = BENCHMARK_REPETITIONS;
        runBenchmarkSuite(perm, scratch, cfg);
        arenaFree(perm); arenaFree(scratch);
        return 0;
    }

    int dims[] = {cfg.inputSize, g_trainConfig.hiddenSize, cfg.outputSize};
    Model *model = modelCreate(perm, dims, NUM_DIMS);

    if (argc > 1 && strcmp(argv[1], "run") == 0) modelLoad(model, cfg.saveFile);
    else trainModel(model, cfg, scratch);

    testPerfect(model, cfg, scratch);
    testRobustness(model, cfg, scratch);
    visualDemo(model, cfg, scratch); 
    displayConfusionMatrix(model, cfg, scratch);

    arenaFree(perm); arenaFree(scratch);
    return 0;
}

void trainModel(Model *model, DatasetConfig cfg, Arena *scratch) {
    printf("--- TRAINING PHASE (%s) ---\n", cfg.name);
    float lr = g_trainConfig.learningRate;
    int indices[cfg.outputSize];
    for(int i=0; i<cfg.outputSize; i++) indices[i] = i;

    for (int pass = 0; pass < TOTAL_PASSES; pass++) {
        shuffle(indices, cfg.outputSize);
        for (int i = 0; i < cfg.outputSize; i++) {
            arenaReset(scratch);
            int idx = indices[i];
            float *sample = cfg.data + (idx * cfg.inputSize);
            glueTrainDigit(model, sample, idx, lr, TRAIN_NOISE, scratch);
        }

        if (pass > 0 && pass % (DECAY_STEP / cfg.outputSize) == 0) {
            lr *= DECAY_RATE;

            arenaReset(scratch);
            int testIdx = rand() % cfg.outputSize;
            Tensor *input = tensorAlloc(scratch, cfg.inputSize, 1);
            memcpy(input->data, cfg.data + (testIdx * cfg.inputSize), cfg.inputSize * sizeof(float));
            Tensor *output = glueForward(model, input, scratch);
            float loss = glueComputeLoss(output, testIdx, scratch);

            printf("Pass %d | Loss: %.6f | LR: %f\n", pass, loss, lr);
        }
    }
    modelSave(model, cfg.saveFile);
}

void testPerfect(Model *model, DatasetConfig cfg, Arena *scratch) {
    printf("\n--- TEST 1: PERFECT SAMPLES ---\n");
    int correct = 0;
    for (int t = 0; t < cfg.outputSize; t++) {
        arenaReset(scratch);
        Tensor *input = tensorAlloc(scratch, cfg.inputSize, 1);
        memcpy(input->data, cfg.data + (t * cfg.inputSize), cfg.inputSize * sizeof(float));
        float conf;
        int guess = gluePredict(model, input, scratch, &conf);
        if (guess == t) correct++;
        printf("Real: %c | AI: %c (Confidence: %.2f%%)\n", cfg.map[t], cfg.map[guess], conf * 100);
    }
    printf("Accuracy: %d/%d\n", correct, cfg.outputSize);
}

void testRobustness(Model *model, DatasetConfig cfg, Arena *scratch) {
    printf("\n--- TEST 2: STRESS TEST (SALT & PEPPER) (%d SAMPLES) ---\n", STRESS_TRIALS);
    int correct = 0;
    int noiseLevel = (cfg.gridSide == 8) ? 4 : 2;
    for (int i = 0; i < STRESS_TRIALS; i++) {
        arenaReset(scratch);
        int label = rand() % cfg.outputSize;
        Tensor *input = tensorAlloc(scratch, cfg.inputSize, 1);
        memcpy(input->data, cfg.data + (label * cfg.inputSize), cfg.inputSize * sizeof(float));
        for(int n=0; n<noiseLevel; n++) 
            input->data[rand() % cfg.inputSize] = 1.0f - input->data[rand() % cfg.inputSize];

        if (gluePredict(model, input, scratch, NULL) == label) correct++;
    }
    printf("Robustness Score (%d-pixel noise): %.2f%%\n", noiseLevel, (float)correct / STRESS_TRIALS * 100);
}

void visualDemo(Model *model, DatasetConfig cfg, Arena *scratch) {
    int targetIdx = rand() % cfg.outputSize;
    printf("\n--- VISUAL DEMO: BROKEN '%c' ---\n", cfg.map[targetIdx]);
    arenaReset(scratch);
    
    Tensor *noisy = tensorAlloc(scratch, cfg.inputSize, 1);
    memcpy(noisy->data, cfg.data + (targetIdx * cfg.inputSize), cfg.inputSize * sizeof(float));
    
    noisy->data[0] = 1.0f - noisy->data[0];
    noisy->data[cfg.inputSize-1] = 1.0f - noisy->data[cfg.inputSize-1];
    noisy->data[cfg.inputSize/2] = 1.0f - noisy->data[cfg.inputSize/2];

    float conf;
    int guess = gluePredict(model, noisy, scratch, &conf);
    
    printDigit(noisy->data, cfg.gridSide);
    printf("AI Guess: %c (Confidence: %.2f%%)\n", cfg.map[guess], conf * 100);
}

void displayConfusionMatrix(Model *model, DatasetConfig cfg, Arena *scratch) {
    printf("\n--- CONFUSION MATRIX ---\n");
    int matrix[cfg.outputSize][cfg.outputSize];
    memset(matrix, 0, sizeof(matrix));

    for (int i = 0; i < CONFUSION_TESTS; i++) {
        arenaReset(scratch);
        int label = rand() % cfg.outputSize;
        Tensor *input = tensorAlloc(scratch, cfg.inputSize, 1);
        for(int j=0; j<cfg.inputSize; j++) {
            float val = (cfg.data + (label * cfg.inputSize))[j];
            if ((rand() % 100) < 5) val = 1.0f - val; 
            input->data[j] = val;
        }
        int guess = gluePredict(model, input, scratch, NULL);
        matrix[label][guess]++;
    }

    printf("R \\ A | ");
    for(int i=0; i<cfg.outputSize; i++) printf("%c ", cfg.map[i]);
    printf("\n-------");
    for(int i=0; i<cfg.outputSize; i++) printf("--");
    printf("\n");

    for (int i = 0; i < cfg.outputSize; i++) {
        printf("  %c   | ", cfg.map[i]);
        for (int j = 0; j < cfg.outputSize; j++) printf("%d ", matrix[i][j]);
        printf("\n");
    }
}

void runBenchmarkSuite(Arena *perm, Arena *scratch, DatasetConfig cfg) {
    printf("--- SCIENTIFIC AI BENCHMARK (N=%d) [%s] ---\n", g_trainConfig.benchmarkReps, cfg.name);
    printf("Hidden |  LR   |  Avg Score  |  Std Dev  | Status\n");
    printf("-------|-------|-------------|-----------|--------\n");

    float bestAvgScore = 0; int bestH = 0; float bestL = 0;

    for (int h = 16; h <= 256; h += 16) {
        float lrs[] = {0.001f, 0.005f, 0.008f, 0.01f, 0.015f, 0.02f};
        for (int l = 0; l < 6; l++) {
            float scores[16]; // max benchmark reps
            float sum = 0;
            int reps = g_trainConfig.benchmarkReps;
            for (int r = 0; r < reps; r++) {
                arenaReset(perm); 
                scores[r] = runExperiment(h, lrs[l], perm, scratch, cfg);
                sum += scores[r];
            }
            float avg = sum / reps;
            float sumSqDiff = 0;
            for (int r = 0; r < reps; r++) sumSqDiff += (scores[r] - avg) * (scores[r] - avg);
            float stdDev = sqrt(sumSqDiff / reps);

            const char* status = "";
            if (avg > bestAvgScore) {
                bestAvgScore = avg; bestH = h; bestL = lrs[l];
                status = (stdDev < 2.0f) ? " <-- STABLE" : " <-- UNSTABLE";
            }
            printf("  %3d  | %.3f |   %6.2f%%  |  %6.2f   | %s\n", h, lrs[l], avg, stdDev, status);
        }
        printf("-------|-------|------------|-----------|--------\n");
    }
    printf("\nWINNER: Hidden=%d, LR=%.3f (Avg: %.2f%%)\n", bestH, bestL, bestAvgScore);
    applyBestParameters(bestH, bestL, cfg);
}

float runExperiment(int hiddenSize, float initialLR, Arena *perm, Arena *scratch, DatasetConfig cfg) {
    int dims[] = {cfg.inputSize, hiddenSize, cfg.outputSize};
    Model *model = modelCreate(perm, dims, NUM_DIMS);
    float lr = initialLR;
    int indices[cfg.outputSize];
    for(int i=0; i<cfg.outputSize; i++) indices[i] = i;

    for (int pass = 0; pass < TOTAL_PASSES; pass++) {
        shuffle(indices, cfg.outputSize);
        for (int i = 0; i < cfg.outputSize; i++) {
            arenaReset(scratch);
            glueTrainDigit(model, cfg.data + (indices[i] * cfg.inputSize), indices[i], lr, TRAIN_NOISE, scratch);
        }
        if (pass > 0 && pass % (DECAY_STEP / cfg.outputSize) == 0) lr *= DECAY_RATE;
    }

    int correct = 0;
    for (int i = 0; i < STRESS_TRIALS; i++) {
        arenaReset(scratch);
        int label = rand() % cfg.outputSize;
        Tensor *input = tensorAlloc(scratch, cfg.inputSize, 1);
        memcpy(input->data, cfg.data + (label * cfg.inputSize), cfg.inputSize * sizeof(float));
        int noiseLevel = (cfg.gridSide == 8) ? 4 : 2;
        for(int n=0; n<noiseLevel; n++) input->data[rand() % cfg.inputSize] = 1.0f - input->data[rand() % cfg.inputSize];
        if (gluePredict(model, input, scratch, NULL) == label) correct++;
    }
    return (float)correct / STRESS_TRIALS * 100.0f;
}

void applyBestParameters(int bestH, float bestL, DatasetConfig cfg) {
    g_trainConfig.hiddenSize = bestH;
    g_trainConfig.learningRate = bestL;

    FILE *f = NULL;
    if (strcmp(cfg.name, "DIGITS") == 0) f = fopen("IO/confs/best_config_DIGITS.txt", "w");
    else if (strcmp(cfg.name, "ALPHANUMERIC") == 0) f = fopen("IO/confs/best_config_ALPHA.txt", "w");
    
    if (!f) { printf("Error: Could not open file to save optimized parameters for %s.\n", cfg.name); return; }

    if (f) {
        fprintf(f, "%d\n%f", bestH, bestL); fclose(f);
        printf("\nOptimized parameters saved in 'IO/confs/best_config_%s.txt'\n", strcmp(cfg.name, "DIGITS") == 0 ? "DIGITS" : "ALPHA");
    }
}

void loadBestParameters(DatasetConfig cfg) {
    FILE *f;
    if(strcmp(cfg.name, "DIGITS") == 0) f = fopen("IO/confs/best_config_DIGITS.txt", "r");
    else f = fopen("IO/confs/best_config_ALPHA.txt", "r");

    if (!f) { printf("Warning: Optimized parameter file not found for %s. Using defaults.\n", cfg.name); return; }

    if (f) {
        if (fscanf(f, "%d\n%f", &g_trainConfig.hiddenSize, &g_trainConfig.learningRate) == 2) 
            printf("Optimized parameters loaded: Hidden=%d, LR=%.3f\n", g_trainConfig.hiddenSize, g_trainConfig.learningRate);
        fclose(f);
    }
}