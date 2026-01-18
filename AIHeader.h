#ifndef AI_HEADER_H
#define AI_HEADER_H

// Core ADTs
#include "headers/Arena.h"
#include "headers/Tensor.h"
#include "headers/Grad.h"
#include "headers/Model.h"
#include "headers/Glue.h"
#include "headers/Utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

extern int   CURRENT_HIDDEN;
extern float CURRENT_LR;

// --- Hyperparameters ---
#define INPUT_SIZE      64

#define DEFAULT_HIDDEN  192     // neurons

#define OUTPUT_SIZE     62
#define NUM_LAYERS      3

// --- Training Parameters ---
#define DEFAULT_LR      0.008f

#define TOTAL_EPOCHS    20000
#define TOTAL_PASSES    2000
#define DECAY_STEP      5000
#define DECAY_RATE      0.7f
#define TRAIN_NOISE     0.10f

// --- Testing Parameters ---
#define STRESS_TRIALS   1000
#define STRESS_NOISE    2      // pixels to flip
#define CONFUSION_TESTS 500

// --- Benchmarking Parameters ---
#define BENCHMARK_REPETITIONS 10

extern float digits[10][25];

#endif