#include "../../AIHeader.h"

/*
 * Provides the definition of g_trainConfig for the test binary.
 * In the main binary this lives in miniAI.c, but the test binary
 * does not link miniAI.o (it has its own main()), so we define it here.
 */
TrainingConfig g_trainConfig = {
    .hiddenSize    = DEFAULT_HIDDEN,
    .learningRate  = DEFAULT_LR,
    .benchmarkReps = BENCHMARK_REPETITIONS,
    .verbose       = 0,
    .seed          = 0
};

// this is rather hacky. Should be refactored in the future.