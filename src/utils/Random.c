#include <stdlib.h>
#include <time.h>
#include "Random.h"

void set_random_seed(unsigned int seed) {
    srand(seed);
}

void randomize() {
    srand((unsigned int)time(NULL));
}