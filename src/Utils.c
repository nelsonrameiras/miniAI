#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper: Renders gridSide x gridSide grid
void printDigit(float *data, int gridSide) {
    int totalPixels = gridSide * gridSide;
    for (int i = 0; i < totalPixels; i++) {
        printf("%s", data[i] > 0.5f ? "## " : ".. ");

        if ((i + 1) % gridSide == 0) printf("\n");
    }
}

void shuffle(int *array, int n) {
    if (n > 1) {
        for (int i = 0; i < n - 1; i++) {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}