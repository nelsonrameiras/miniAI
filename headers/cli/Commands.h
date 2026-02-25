#ifndef _COMMANDS_H
#define _COMMANDS_H

#include "ArgParse.h"
#include "../core/Arena.h"

// Command execution functions
int cmdTrain(const CommandArgs *args);
int cmdTest(const CommandArgs *args);
int cmdBenchmark(const CommandArgs *args);
int cmdRecognize(const CommandArgs *args);

// Command dispatcher
int executeCommand(const CommandArgs *args);

#endif // _COMMANDS_H