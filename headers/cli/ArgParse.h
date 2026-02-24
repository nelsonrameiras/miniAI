#ifndef ARGPARSE_H
#define ARGPARSE_H

// Command types
typedef enum {
    CMD_TRAIN,
    CMD_TEST,
    CMD_BENCHMARK,
    CMD_RECOGNIZE,
    CMD_HELP,
    CMD_VERSION,
    CMD_INVALID
} CommandType;

// Dataset specification
typedef enum {
    DATASET_SPEC_DIGITS,
    DATASET_SPEC_ALPHA,
    DATASET_SPEC_CUSTOM
} DatasetSpec;

// Parsed command-line arguments
typedef struct {
    CommandType command;
    DatasetSpec dataset;
    
    const char *dataPath;      // Path to data (directory for PNG, file for phrase)
    const char *modelFile;     // Path to model file
    const char *imageFile;     // Path to image file (for test/recognize)
    const char *configFile;    // Path to config file
    
    int gridSize;              // Grid size (0 = auto-detect)
    int benchmarkReps;         // Benchmark repetitions (default 3)
    int loadModel;             // Load existing model (1) or train (0)
    int resumeModel;           // Load existing model and continue training (1)
    int useStatic;             // Use static in-memory dataset (1) or PNG (0)
    int verbose;               // Verbose output
    int seed;                  // Random seed (0 = random, >0 = fixed for reproducibility)
} CommandArgs;

// Parse command-line arguments
CommandArgs parseArgs(int argc, char **argv);

// Print usage information
void printUsage(const char *progName);

// Get default paths for dataset
void getDefaultPaths(DatasetSpec spec, int useStatic, const char **dataPath, 
                    const char **modelFile, int *gridSize);

#endif // ARGPARSE_H