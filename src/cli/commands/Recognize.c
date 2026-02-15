#include "../../headers/cli/Commands.h"
#include "../../headers/dataset/Dataset.h"
#include "../../headers/core/Model.h"
#include "../../headers/core/Glue.h"
#include "../../AIHeader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern TrainingConfig g_trainConfig;

int cmdRecognize(CommandArgs args) {
    printf("=== PHRASE RECOGNITION MODE ===\n\n");
    
    if (!args.imageFile) {
        fprintf(stderr, "Error: recognize command requires --image option\n");
        return 1;
    }
    
    if (!args.modelFile) {
        fprintf(stderr, "Error: recognize command requires --model option\n");
        return 1;
    }
    
    // Create arenas
    Arena *perm = arenaInit(8 * MB);
    Arena *scratch = arenaInit(2 * MB);
    
    if (!perm || !scratch) {
        fprintf(stderr, "Error: Could not allocate memory\n");
        return 1;
    }
    
    // Determine character map and output size
    const char *charMap;
    int outputSize;
    
    if (args.dataset == DATASET_SPEC_DIGITS) {
        charMap = CHARMAP_DIGITS;
        outputSize = 10;
    } else {
        charMap = CHARMAP_ALPHA;
        outputSize = 62;
    }
    
    int inputSize = args.gridSize * args.gridSize;
    
    // Create phrase dataset (segments the image)
    printf("Loading and segmenting phrase from: %s\n", args.imageFile);
    Dataset *ds = datasetCreatePhrase(args.imageFile, args.gridSize, args.modelFile, charMap);
    
    if (!ds) {
        fprintf(stderr, "Error: Could not load or segment phrase\n");
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    printf("Segmented %d characters\n", ds->phrase.sequence->count);
    printf("Grid: %dx%d (%d inputs per character)\n\n", args.gridSize, args.gridSize, inputSize);
    
    // Create and load model
    int dims[] = {inputSize, g_trainConfig.hiddenSize, outputSize};
    Model *model = modelCreate(perm, dims, NUM_DIMS);
    
    if (!model) {
        fprintf(stderr, "Error: Could not create model\n");
        datasetFree(ds);
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    printf("Loading model from %s\n", args.modelFile);
    if (modelLoad(model, args.modelFile) != 0) {
        fprintf(stderr, "Error: Could not load model\n");
        datasetFree(ds);
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    // Recognize each character
    char *phrase = (char*)malloc(ds->phrase.sequence->count + 1);
    float *confidences = (float*)malloc(ds->phrase.sequence->count * sizeof(float));
    
    if (!phrase || !confidences) {
        fprintf(stderr, "Error: Could not allocate memory for results\n");
        datasetFree(ds);
        arenaFree(perm);
        arenaFree(scratch);
        return 1;
    }
    
    printf("\n========================================\n");
    printf("         RECOGNIZED PHRASE\n");
    printf("========================================\n\n");
    
    int phraseLen = 0;
    for (int i = 0; i < ds->phrase.sequence->count; i++) {
        arenaReset(scratch);
        
        // Check for space
        if (ds->phrase.sequence->chars[i] == NULL) {
            phrase[phraseLen++] = ' ';
            confidences[i] = 0;
            continue;
        }
        
        // Create input tensor
        Tensor *input = tensorAlloc(scratch, inputSize, 1);
        memcpy(input->data, ds->phrase.sequence->chars[i], inputSize * sizeof(float));
        
        // Predict
        float conf;
        int prediction = gluePredict(model, input, scratch, &conf);
        
        phrase[phraseLen++] = charMap[prediction];
        confidences[i] = conf;
    }
    phrase[phraseLen] = '\0';
    
    // Display result
    printf("  \"%s\"\n\n", phrase);
    
    // Show character details
    printf("--- Character Details ---\n");
    printf("Pos | Char | Confidence\n");
    printf("----|------|------------\n");
    
    for (int i = 0; i < ds->phrase.sequence->count; i++) {
        if (ds->phrase.sequence->chars[i] == NULL) {
            printf("%3d |       | (space)\n", i);
        } else {
            printf("%3d |  %c   |   %.2f%%\n", i, phrase[i == 0 ? 0 : i], confidences[i] * 100);
        }
    }
    
    printf("\n========================================\n");
    
    // Cleanup
    free(phrase);
    free(confidences);
    datasetFree(ds);
    arenaFree(perm);
    arenaFree(scratch);
    
    return 0;
}