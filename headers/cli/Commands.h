#ifndef _COMMANDS_H
#define _COMMANDS_H

#include "ArgParse.h"
#include "Arena.h"

// Command execution functions
int cmdTrain(CommandArgs args);
int cmdTest(CommandArgs args);
int cmdBenchmark(CommandArgs args);
int cmdRecognize(CommandArgs args);

// Command dispatcher
int executeCommand(CommandArgs args);

#endif // _COMMANDS_H