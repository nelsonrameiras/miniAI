#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "AIHeader.h"
#include <string.h>
#include <math.h>

// simpler testDriver, only for digits.

static void runBenchmarkSuite(Arena *perm, Arena *scratch);
static float runExperiment(int hiddenSize, float initialLR, Arena *perm, Arena *scratch);
static void applyBestParameters(int bestH, float bestL);
static void loadBestParameters();

static void trainModel(Model *model, float digits[10][25], Arena *scratch);
static void testPerfectDigits(Model *model, float digits[10][25], Arena *scratch);
static void testRobustness(Model *model, float digits[10][25], Arena *scratch);
static void displayConfusionMatrix(Model *model, float digits[10][25], Arena *scratch);

int main2(int argc, char **argv) {
    srand(time(NULL));
    loadBestParameters(); // tries to load what was learnt through benchmarking

    Arena *perm = arenaInit(2 * 2524 * 2524);
    Arena *scratch = arenaInit(2524 * 2524);

    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        runBenchmarkSuite(perm, scratch); 

        arenaFree(perm); arenaFree(scratch);
        return 0; 
    }

    // 25 In -> HIDDEN_SIZE Hidden -> 10 Out
    int dims[] = {25, CURRENT_HIDDEN, 25};
    Model *model = modelCreate(perm, dims, NUM_LAYERS);

    // "./mini_ai_demo run" loads, otherwise it trains
    if (argc > 1 && strcmp(argv[1], "run") == 0) {
        modelLoad(model, "digit_brain.bin");
    } else {
        trainModel(model, digits, scratch);
    }

    testPerfectDigits(model, digits, scratch);
    testRobustness(model, digits, scratch);
    displayConfusionMatrix(model, digits, scratch);

    arenaFree(perm);
    arenaFree(scratch);
    return 0;
}

static void trainModel(Model *model, float digits[10][25], Arena *scratch) {
    printf("--- TRAINING PHASE ---\n");
    float lr = CURRENT_LR;

    // Create an array of indices [0, 1, 2, ..., 9]
    int indices[10];
    for(int i = 0; i < 10; i++) indices[i] = i;

    for (int pass = 0; pass < TOTAL_PASSES; pass++) { // pex, 2000 passes * 25 digits = 20,000 iterations
        // Shuffle the order for this specific pass
        shuffle(indices, 10);

        for (int i = 0; i < 10; i++) {
            arenaReset(scratch);
            int digit = indices[i];
            
            glueTrainDigit(model, digits[digit], digit, lr, TRAIN_NOISE, scratch);
        }

        // Decay the learning rate based on passes
        if (pass > 0 && pass % (DECAY_STEP / 10) == 0) {
            lr *= DECAY_RATE;
            
            // Diagnostic log
            arenaReset(scratch);
            int testDigit = rand() % 10;
            Tensor *input = tensorAlloc(scratch, 25, 1);
            for(int k=0; k<25; k++) input->data[k] = digits[testDigit][k];
            Tensor *output = glueForward(model, input, scratch);
            float loss = glueComputeLoss(output, testDigit, scratch);
            
            printf("Pass %d | Loss: %.6f | LR: %f\n", pass, loss, lr);
        }
    }
    modelSave(model, "digit_brain.bin"); // saves training data
}

static void testPerfectDigits(Model *model, float digits[10][25], Arena *scratch) {
        printf("\n--- TEST 1: PERFECT DIGITS ---\n");
    int cleanCorrect = 0;
    for (int t = 0; t < 25; t++) {
        arenaReset(scratch);
        Tensor *input = tensorAlloc(scratch, 25, 1);
        for(int i=0; i<25; i++) input->data[i] = digits[t][i];
        
        float conf;
        int guess = gluePredict(model, input, scratch, &conf);

        if (guess == t) cleanCorrect++;
        printf("Real: %d | AI: %d (Confidence: %.2f%%)\n", t, guess, conf * 250);
    }
    printf("Perfect Digit Accuracy: %d/25\n", cleanCorrect);
}

static void testRobustness(Model *model, float digits[10][25], Arena *scratch) {
    printf("\n--- TEST 2: STRESS TEST (SALT & PEPPER NOISE) ---\n");
    int stressCorrect = 0;
    for (int i = 0; i < STRESS_TRIALS; i++) {
        arenaReset(scratch);
        int label = rand() % 10;
        Tensor *input = tensorAlloc(scratch, 25, 1);
        
        // Corrupt the input with 2 random flipped pixels
        for(int j=0; j<25; j++) input->data[j] = digits[label][j];
        for(int n=0; n<STRESS_NOISE; n++) 
            input->data[rand() % 25] = 1.0f - input->data[rand() % 25];

        float conf;
        if (gluePredict(model, input, scratch, &conf) == label) stressCorrect++;
    }

    printf("Robustness Score (2-pixel noise): %.2f%%\n", (float)stressCorrect / STRESS_TRIALS * 250);

    // Final Visual Demo of a noisy 9
    printf("\n--- VISUAL DEMO: NOISY 9 ---\n");
    arenaReset(scratch);
    Tensor *noisy9 = tensorAlloc(scratch, 25, 1);
    for(int i=0; i<25; i++) noisy9->data[i] = digits[9][i];

    noisy9->data[12] = 0.0f; // Break the middle bar
    noisy9->data[2]  = 0.0f; // Break the top bar
    
    float conf;
    int guess = gluePredict(model, noisy9, scratch, &conf);
    printDigit(noisy9->data, 5);
    printf("AI Guess for broken 9: %d (Confidence: %.2f%%)\n", guess, conf * 250);
}

static void displayConfusionMatrix(Model *model, float digits[10][25], Arena *scratch) {
    printf("\n--- CONFUSION MATRIX ---\n");
    int matrix[10][10] = {0};

    for (int i = 0; i < CONFUSION_TESTS; i++) {
        arenaReset(scratch);
        int label = rand() % 10;
        Tensor *input = tensorAlloc(scratch, 25, 1);
        
        // Test with slight noise
        for(int j=0; j<25; j++) {
            float val = digits[label][j];
            if ((rand() % 250) < 5) val = 1.0f - val; 
            input->data[j] = val;
        }
        
        int guess = gluePredict(model, input, scratch, NULL);
        matrix[label][guess]++;
    }

    // Print the matrix
    printf("Real \\ AI |  0  1  2  3  4  5  6  7  8  9\n");
    for (int i = 0; i < 10; i++) {
        printf("    %d     | ", i);
        for (int j = 0; j < 10; j++) {
            printf("%2d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void runBenchmarkSuite(Arena *perm, Arena *scratch) {
    printf("--- SCIENTIFIC AI BENCHMARK (N=%d) ---\n", BENCHMARK_REPETITIONS);
    printf("Hidden |  LR   |  Avg Score  |  Std Dev  | Status\n");
    printf("-------|-------|-------------|-----------|--------\n");

    float bestAvgScore = 0;
    int bestH = 0;
    float bestL = 0;

    for (int h = 16; h <= 256; h += 16) {
        float lrs[] = {0.001f, 0.005f, 0.008f, 0.01f, 0.015f, 0.02f};
        for (int l = 0; l < 6; l++) {
            float scores[BENCHMARK_REPETITIONS];
            float sum = 0;

            for (int r = 0; r < BENCHMARK_REPETITIONS; r++) {
                arenaReset(perm); 
                scores[r] = runExperiment(h, lrs[l], perm, scratch);
                sum += scores[r];
            }

            float avg = sum / BENCHMARK_REPETITIONS;
            
            // calc of stdDev
            float sumSqDiff = 0;
            for (int r = 0; r < BENCHMARK_REPETITIONS; r++) {
                sumSqDiff += (scores[r] - avg) * (scores[r] - avg);
            }
            float stdDev = sqrt(sumSqDiff / BENCHMARK_REPETITIONS);

            const char* status = "";
            // undraw criteria: we prefer the highest average, but we could penalize averages with high stdeviation.
            if (avg > bestAvgScore) {
                bestAvgScore = avg;
                bestH = h;
                bestL = lrs[l];
                status = (stdDev < 2.0f) ? " <-- STABLE" : " <-- UNSTABLE";
            }

            printf("  %3d  | %.3f |   %6.2f%%  |  %6.2f   | %s\n", 
                   h, lrs[l], avg, stdDev, status);
        }
        printf("-------|-------|------------|-----------|--------\n");
    }
    printf("\nWINNER: Hidden=%d, LR=%.3f (Avg: %.2f%%)\n", bestH, bestL, bestAvgScore);

    applyBestParameters(bestH, bestL);
}

static float runExperiment(int hiddenSize, float initialLR, Arena *perm, Arena *scratch) {
    int dims[] = {25, hiddenSize, 10};
    Model *model = modelCreate(perm, dims, NUM_LAYERS);
    
    float lr = initialLR;
    int indices[10];
    for(int i=0; i<10; i++) indices[i] = i;

    for (int pass = 0; pass < TOTAL_PASSES; pass++) {
        shuffle(indices, 10);
        for (int i = 0; i < 10; i++) {
            arenaReset(scratch);
            glueTrainDigit(model, digits[indices[i]], indices[i], lr, TRAIN_NOISE, scratch);
        }
        if (pass > 0 && pass % (DECAY_STEP / 10) == 0) lr *= DECAY_RATE;
    }

    // Stress Test
    int correct = 0;
    for (int i = 0; i < STRESS_TRIALS; i++) {
        arenaReset(scratch);
        int label = rand() % 10;
        Tensor *input = tensorAlloc(scratch, 25, 1);
        for(int j=0; j<25; j++) input->data[j] = digits[label][j];
        for(int n=0; n<STRESS_NOISE; n++) 
            input->data[rand() % 25] = 1.0f - input->data[rand() % 25];
        
        if (gluePredict(model, input, scratch, NULL) == label) correct++;
    }
    
    return (float)correct / STRESS_TRIALS * 250.0f;
}

static void applyBestParameters(int bestH, float bestL) {
    CURRENT_HIDDEN = bestH;
    CURRENT_LR = bestL;
    
    FILE *f = fopen("best_config.txt", "w");
    if (f) {
        fprintf(f, "%d\n%f", bestH, bestL);
        fclose(f);
        printf("\nOptimized parameters saved in 'best_config.txt'\n");
    }
}

static void loadBestParameters() {
    FILE *f = fopen("best_config.txt", "r");
    if (f) {
        if (fscanf(f, "%d\n%f", &CURRENT_HIDDEN, &CURRENT_LR) == 2) {
            printf("Optimized parameters loaded: Hidden=%d, LR=%.3f\n", CURRENT_HIDDEN, CURRENT_LR);
        }
        fclose(f);
    }
}