#include "AIHeader.h"
#include "headers/cli/ArgParse.h"
#include "headers/cli/Commands.h"
#include <stdio.h>
#include <stdlib.h>
#include "headers/utils/Random.h"

// Global training configuration
TrainingConfig g_trainConfig;

int main(int argc, char **argv) {
    // Initialize default training configuration
    g_trainConfig = defaultTrainingConfig();
    
    // Parse command-line arguments
    CommandArgs args = parseArgs(argc, argv);

    // Apply verbose and seed from args to global config
    g_trainConfig.verbose = args.verbose;
    g_trainConfig.seed    = args.seed;
    
    // Initialize random seed
    if (args.seed > 0) {
        set_random_seed((unsigned int)args.seed);
        if (args.verbose) printf("Random seed: %d\n", args.seed);
    } else randomize();

    // Execute command
    int result = executeCommand(args);
    
    return result;
}