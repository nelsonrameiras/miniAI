#include "AIHeader.h"
#include "headers/cli/ArgParse.h"
#include "headers/cli/Commands.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Global training configuration
TrainingConfig g_trainConfig;

int main(int argc, char **argv) {
    // Initialize random seed
    srand(time(NULL));
    
    // Initialize default training configuration
    g_trainConfig = defaultTrainingConfig();
    
    // Parse command-line arguments
    CommandArgs args = parseArgs(argc, argv);
    
    // Execute command
    int result = executeCommand(args);
    
    return result;
}