#include "../../headers/cli/Commands.h"
#include "../../headers/cli/ArgParse.h"
#include <stdio.h>

int executeCommand(CommandArgs args) {
    switch(args.command) {
        case CMD_TRAIN:
            return cmdTrain(args);
            
        case CMD_TEST:
            return cmdTest(args);
            
        case CMD_BENCHMARK:
            return cmdBenchmark(args);
            
        case CMD_RECOGNIZE:
            return cmdRecognize(args);
            
        case CMD_HELP:
            printUsage("miniAI");
            return 0;
            
        case CMD_INVALID:
        default:
            fprintf(stderr, "Error: Invalid command\n\n");
            printUsage("miniAI");
            return 1;
    }
}