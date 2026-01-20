#ifndef AI_HEADER_H
#define AI_HEADER_H

// Core ADTs
#include "headers/Arena.h"
#include "headers/Tensor.h"
#include "headers/Grad.h"
#include "headers/Model.h"
#include "headers/Glue.h"
#include "headers/Utils.h"
#include "headers/ImageLoader.h"
#include "headers/ImagePreprocess.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>

// --- Network Architecture Defaults ---
#define DEFAULT_HIDDEN  1024     // default hidden layer neurons

// NUM_DIMS = number of dimension values (input, hidden, output)
// Actual layer count = NUM_DIMS - 1 (2 layers for a 3-dim network)
#define NUM_DIMS        3

// --- Training Parameters ---
#define DEFAULT_LR      0.005f
#define LAMBDA          0.0001f  // L2 regularization factor
#define GRAD_CLIP       1.0f     // Gradient clipping threshold

#define TOTAL_PASSES    2000
#define DECAY_STEP      5000
#define DECAY_RATE      0.7f
#define TRAIN_NOISE     0.10f

// --- Testing Parameters ---
#define STRESS_TRIALS   1000
#define STRESS_NOISE    2      // pixels to flip
#define CONFUSION_TESTS 500

// --- Benchmarking Parameters ---
#define BENCHMARK_REPETITIONS 1

// --- Training Configuration ---
// Holds runtime-adjustable training parameters
typedef struct {
    int   hiddenSize;
    float learningRate;
    int   benchmarkReps;
} TrainingConfig;

// Global training config instance (replaces individual globals)
extern TrainingConfig g_trainConfig;

// Helper to initialize default training config
static inline TrainingConfig defaultTrainingConfig(void) {
    return (TrainingConfig){
        .hiddenSize = DEFAULT_HIDDEN,
        .learningRate = DEFAULT_LR,
        .benchmarkReps = BENCHMARK_REPETITIONS
    };
}

extern float digits[10][25];

#endif