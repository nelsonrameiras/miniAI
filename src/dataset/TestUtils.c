#include "../headers/dataset/TestUtils.h"
#include "../headers/core/Glue.h"
#include "../headers/Utils.h"
#include "../AIHeader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern TrainingConfig g_trainConfig;

// ===== TRAINING =====

void trainModel(Model *model, Dataset *ds, Arena *scratch) {
    printf("--- TRAINING PHASE (%s) ---\n", ds->name);
    float lr = g_trainConfig.learningRate;
    
    int *indices = (int*)malloc(ds->outputSize * sizeof(int));
    if (!indices) { fprintf(stderr, "Error: Could not allocate indices array\n"); return; }
    
    for(int i = 0; i < ds->outputSize; i++) indices[i] = i;

    for (int pass = 0; pass < TOTAL_PASSES; pass++) {
        shuffle(indices, ds->outputSize);

        int currentBatchSize = 0;
        
        for (int i = 0; i < ds->outputSize; i++) {
            arenaReset(scratch);
            int idx = indices[i];
            
            float *sample = datasetGetSample(ds, idx);

            if (sample) {
                // Instead of training right away, only accumulate the gradient
                glueAccumulateGradients(model, sample, idx, TRAIN_NOISE, scratch);
                currentBatchSize++;
            }
            
            if (currentBatchSize == BATCH_SIZE || i == ds->outputSize - 1) {
                if (currentBatchSize > 0) {
                    glueUpdateWeights(model, lr, currentBatchSize);
                    currentBatchSize = 0;
                }
            }
        }

        // Learning rate decay and diagnostic logging
        if (pass > 0 && pass % (DECAY_STEP ) == 0) {
            lr *= DECAY_RATE;
            
            // Compute diagnostic loss
            arenaReset(scratch);
            int testIdx = rand() % ds->outputSize;
            Tensor *input = tensorAlloc(scratch, ds->inputSize, 1);
            
            float *sample = datasetGetSample(ds, testIdx);
            if (sample) {
                memcpy(input->data, sample, ds->inputSize * sizeof(float));
                Tensor *output = glueForward(model, input, scratch);
                float loss = glueComputeLoss(output, testIdx, scratch);
                printf("Pass %d | Loss: %.6f | LR: %f\n", pass, loss, lr);
            }
        }
    }
    
    free(indices);
    modelSave(model, ds->saveFile);
}

// ===== TESTING =====

void testPerfect(Model *model, Dataset *ds, Arena *scratch) {
    printf("\n--- TEST 1: PERFECT SAMPLES ---\n");
    int correct = 0;
    int tested = 0;
    
    for (int t = 0; t < ds->outputSize; t++) {
        arenaReset(scratch);
        
        float *sample = datasetGetSample(ds, t);
        if (!sample) continue;
        
        Tensor *input = tensorAlloc(scratch, ds->inputSize, 1);
        memcpy(input->data, sample, ds->inputSize * sizeof(float));
        
        float conf;
        int guess = gluePredict(model, input, scratch, &conf);
        
        if (guess == t) correct++;
        tested++;
        
        printf("Real: %c | AI: %c (Confidence: %.2f%%)\n", 
               ds->charMap[t], ds->charMap[guess], conf * 100);
    }
    
    printf("Accuracy: %d/%d (%.2f%%)\n", correct, tested, 
           tested > 0 ? (float)correct / tested * 100 : 0);
}

void testRobustness(Model *model, Dataset *ds, Arena *scratch) {
    printf("\n--- TEST 2: STRESS TEST (SALT & PEPPER) (%d SAMPLES) ---\n", STRESS_TRIALS);
    int correct = 0;
    
    for (int i = 0; i < STRESS_TRIALS; i++) {
        arenaReset(scratch);
        int label = rand() % ds->outputSize;
        
        float *sample = datasetGetSample(ds, label);
        if (!sample) continue;
        
        Tensor *input = tensorAlloc(scratch, ds->inputSize, 1);
        
        // Copy and add noise
        memcpy(input->data, sample, ds->inputSize * sizeof(float));
        for(int n = 0; n < STRESS_NOISE; n++) {
            int idx = rand() % ds->inputSize;
            input->data[idx] = 1.0f - input->data[idx];
        }
        
        float conf;
        if (gluePredict(model, input, scratch, &conf) == label) {
            correct++;
        }
    }
    
    printf("Robustness Score (%d-pixel noise): %.2f%%\n", 
           STRESS_NOISE, (float)correct / STRESS_TRIALS * 100);
}

void displayConfusionMatrix(Model *model, Dataset *ds, Arena *scratch) {
    printf("\n--- CONFUSION MATRIX ---\n");
    
    // Allocate matrix
    int **matrix = (int**)malloc(ds->outputSize * sizeof(int*));
    for (int i = 0; i < ds->outputSize; i++) {
        matrix[i] = (int*)calloc(ds->outputSize, sizeof(int));
    }
    
    for (int i = 0; i < CONFUSION_TESTS; i++) {
        arenaReset(scratch);
        int label = rand() % ds->outputSize;
        
        float *sample = datasetGetSample(ds, label);
        if (!sample) continue;
        
        Tensor *input = tensorAlloc(scratch, ds->inputSize, 1);
        
        // Test with slight noise
        for(int j = 0; j < ds->inputSize; j++) {
            float val = sample[j];
            if ((rand() % 250) < 5) {
                val = 1.0f - val;
            }
            input->data[j] = val;
        }
        
        int guess = gluePredict(model, input, scratch, NULL);
        matrix[label][guess]++;
    }
    
    // Print matrix header
    printf("Real \\ AI | ");
    for (int j = 0; j < ds->outputSize; j++) {
        printf("%2c ", ds->charMap[j]);
    }
    printf("\n");
    
    // Print matrix rows
    for (int i = 0; i < ds->outputSize; i++) {
        printf("    %c     | ", ds->charMap[i]);
        for (int j = 0; j < ds->outputSize; j++) {
            printf("%2d ", matrix[i][j]);
        }
        printf("\n");
    }
    
    // Cleanup
    for (int i = 0; i < ds->outputSize; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void visualDemo(Model *model, Dataset *ds, Arena *scratch) {
    printf("\n--- VISUAL DEMO ---\n");
    
    // Pick a random sample
    int idx = rand() % ds->outputSize;
    float *sample = datasetGetSample(ds, idx);
    if (!sample) {
        printf("Could not get sample for demo\n");
        return;
    }
    
    arenaReset(scratch);
    Tensor *input = tensorAlloc(scratch, ds->inputSize, 1);
    memcpy(input->data, sample, ds->inputSize * sizeof(float));
    
    // Add some noise
    for (int i = 0; i < 3; i++) {
        int pos = rand() % ds->inputSize;
        input->data[pos] = 1.0f - input->data[pos];
    }
    
    printf("Noisy sample of '%c':\n", ds->charMap[idx]);
    printDigit(input->data, ds->gridSize);
    
    float conf;
    int guess = gluePredict(model, input, scratch, &conf);
    printf("AI Prediction: '%c' (Confidence: %.2f%%)\n", 
           ds->charMap[guess], conf * 100);
}

// ===== BENCHMARKING =====

BenchmarkResult runSingleExperiment(int hiddenSize, float lr, Dataset *ds, 
                                   Arena *perm, Arena *scratch, int repetitions) {
    BenchmarkResult result = {
        .hiddenSize = hiddenSize,
        .learningRate = lr,
        .avgScore = 0,
        .stdDev = 0
    };
    
    float *scores = (float*)malloc(repetitions * sizeof(float));
    if (!scores) return result;
    
    float sum = 0;
    
    for (int rep = 0; rep < repetitions; rep++) {
        arenaReset(perm);
        
        // Create model
        int dims[] = {ds->inputSize, hiddenSize, ds->outputSize};
        Model *model = modelCreate(perm, dims, NUM_DIMS);
        
        // Train
        float learningRate = lr;
        int *indices = (int*)malloc(ds->outputSize * sizeof(int));
        for(int i = 0; i < ds->outputSize; i++) indices[i] = i;
        
        for (int pass = 0; pass < TOTAL_PASSES; pass++) {
            shuffle(indices, ds->outputSize);
            int currentBatchSize = 0;

            for (int i = 0; i < ds->outputSize; i++) {
                arenaReset(scratch);
                float *sample = datasetGetSample(ds, indices[i]);
                if (sample) {
                    glueAccumulateGradients(model, sample, indices[i], TRAIN_NOISE, scratch);
                    currentBatchSize++;
                }

                if (currentBatchSize == BATCH_SIZE || i == ds->outputSize - 1) {
                    if (currentBatchSize > 0) {
                        glueUpdateWeights(model, learningRate, currentBatchSize);
                        currentBatchSize = 0;
                    }
                }
            }
            if (pass > 0 && pass % (DECAY_STEP / ds->outputSize) == 0) {
                learningRate *= DECAY_RATE;
            }
        }
        free(indices);
        
        // Test robustness
        int correct = 0;
        for (int i = 0; i < STRESS_TRIALS; i++) {
            arenaReset(scratch);
            int label = rand() % ds->outputSize;
            float *sample = datasetGetSample(ds, label);
            if (!sample) continue;
            
            Tensor *input = tensorAlloc(scratch, ds->inputSize, 1);
            memcpy(input->data, sample, ds->inputSize * sizeof(float));
            for(int n = 0; n < STRESS_NOISE; n++) {
                input->data[rand() % ds->inputSize] = 1.0f - input->data[rand() % ds->inputSize];
            }
            
            if (gluePredict(model, input, scratch, NULL) == label) {
                correct++;
            }
        }
        
        scores[rep] = (float)correct / STRESS_TRIALS * 100.0f;
        sum += scores[rep];
    }
    
    result.avgScore = sum / repetitions;
    
    // Calculate standard deviation
    float sumSqDiff = 0;
    for (int i = 0; i < repetitions; i++) {
        float diff = scores[i] - result.avgScore;
        sumSqDiff += diff * diff;
    }
    result.stdDev = sqrtf(sumSqDiff / repetitions);
    
    free(scores);
    return result;
}

void runBenchmark(Dataset *ds, Arena *perm, Arena *scratch, int repetitions) {
    printf("--- SCIENTIFIC AI BENCHMARK (N=%d) ---\n", repetitions);
    printf("Dataset: %s\n", ds->name);
    printf("Hidden |  LR   |  Avg Score |  Std Dev  | Status\n");
    printf("-------|-------|------------|-----------|--------\n");
    
    float bestAvgScore = 0;
    int bestH = 0;
    float bestL = 0;
    
    int hiddenSizes[] = {16, 32, 64, 128, 256, 512, 1024};
    float learningRates[] = {0.001f, 0.005f, 0.008f, 0.01f, 0.015f, 0.02f};
    
    int numHidden = sizeof(hiddenSizes) / sizeof(hiddenSizes[0]);
    int numLR = sizeof(learningRates) / sizeof(learningRates[0]);
    
    for (int h = 0; h < numHidden; h++) {
        for (int l = 0; l < numLR; l++) {
            BenchmarkResult result = runSingleExperiment(
                hiddenSizes[h], learningRates[l], ds, perm, scratch, repetitions
            );
            
            const char* status = "";
            if (result.avgScore > bestAvgScore) {
                bestAvgScore = result.avgScore;
                bestH = hiddenSizes[h];
                bestL = learningRates[l];
                status = (result.stdDev < 2.0f) ? " <-- STABLE" : " <-- UNSTABLE";
            }
            
            printf("  %4d | %.3f |   %6.2f%%  |  %6.2f   | %s\n", 
                   hiddenSizes[h], learningRates[l], result.avgScore, result.stdDev, status);
        }
        printf("-------|-------|------------|-----------|--------\n");
    }
    
    printf("\nWINNER: Hidden=%d, LR=%.3f (Avg: %.2f%%)\n", bestH, bestL, bestAvgScore);
    
    // Save best configuration
    char configFile[256];
    const char *datasetType = ds->isStatic ? "static" : "png";
    if (strcmp(ds->name, "DIGITS") == 0) {
        snprintf(configFile, sizeof(configFile), "IO/configs/best_config_digits_%s.txt", datasetType);
    } else {
        snprintf(configFile, sizeof(configFile), "IO/configs/best_config_alpha_%s.txt", datasetType);
    }
    saveBestConfig(configFile, bestH, bestL);
}

// ===== CONFIGURATION MANAGEMENT =====

void loadBestConfig(const char *configFile) {
    FILE *f = fopen(configFile, "r");
    if (f) {
        if (fscanf(f, "%d\n%f", &g_trainConfig.hiddenSize, &g_trainConfig.learningRate) == 2) {
            printf("Loaded config: Hidden=%d, LR=%.3f\n", 
                   g_trainConfig.hiddenSize, g_trainConfig.learningRate);
        }
        fclose(f);
    }
}

void saveBestConfig(const char *configFile, int hiddenSize, float lr) {
    FILE *f = fopen(configFile, "w");
    if (f) {
        fprintf(f, "%d\n%f", hiddenSize, lr);
        fclose(f);
        printf("Saved optimized config to %s\n", configFile);
    } else {
        fprintf(stderr, "Warning: Could not save config to %s\n", configFile);
    }
}