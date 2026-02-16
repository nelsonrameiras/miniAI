#include "../headers/core/Glue.h"
#include "../headers/core/Grad.h"
#include "../AIHeader.h"
#include <stdlib.h>
#include <math.h>

// Diagnostic tool to see how well the model is learning
float glueComputeLoss(Tensor *output, int label, Arena *scratch) {
    // 1. We must apply Softmax to the raw output to get probabilities
    float epsilon = 1e-7f;
    
    Tensor *probs = tensorAlloc(scratch, output->rows, 1); // temp for calculation
    tensorSoftmax(probs, output);
    
    float probOfCorrect = probs->data[label];
    if (probOfCorrect < epsilon) probOfCorrect = epsilon;
    
    return -logf(probOfCorrect); 
}

Tensor* glueForward(Model *m, Tensor *input, Arena *scratch) {
    Tensor *currentInput = input;
    for (int i = 0; i < m->count; i++) {
        m->layers[i].z = tensorAlloc(scratch, m->layers[i].w->rows, 1);
        m->layers[i].a = tensorAlloc(scratch, m->layers[i].w->rows, 1);
        
        tensorDot(m->layers[i].z, m->layers[i].w, currentInput);
        tensorAdd(m->layers[i].z, m->layers[i].z, m->layers[i].b);
        
        // use ReLU for hidden, Softmax logic is handled in training.
        if (i < m->count - 1) {
            tensorReLU(m->layers[i].a, m->layers[i].z);
        } else {
            // for the very last layer, we can use raw for Softmax (we could do it a different way...)
            // Softmax in gluePredict/Train will handle the probabilities.
            for(int j=0; j<m->layers[i].z->rows; j++) 
                m->layers[i].a->data[j] = m->layers[i].z->data[j];
        }
        currentInput = m->layers[i].a;
    }
    return currentInput;
}

int gluePredict(Model *m, Tensor *input, Arena *scratch, float *outConfidence) {
    // initial memory pointer
    size_t startPos = scratch->used;

    Tensor *output = glueForward(m, input, scratch);
    
    Tensor *probs = tensorAlloc(scratch, output->rows, 1);
    tensorSoftmax(probs, output);

    int guess = 0;
    for (int i = 1; i < probs->rows; i++) 
        if (probs->data[i] > probs->data[guess]) guess = i;
    
    if (outConfidence) *outConfidence = probs->data[guess];

    // rewind the memory: implicitly throw away the intermediate tensors gen by glueForward and softmax (memory improvement)
    scratch->used = startPos;

    return guess;
}

void glueAccumulateGradients(Model *m, float *rawData, int label, float noiseLevel, Arena *scratch) {
    // 1. Prepares Input e applies Data Augmentation (Noise)
    Tensor *input = tensorAlloc(scratch, m->layers[0].w->cols, 1);
    for(int i = 0; i < input->rows; i++) {
        float val = rawData[i];
        if (noiseLevel > 0 && ((float)rand()/(float)RAND_MAX) < noiseLevel) val = 1.0f - val; // flip pixel
        input->data[i] = val;
    }

    // 2. Forward pass
    Tensor *output = glueForward(m, input, scratch);
    Tensor *probs = tensorAlloc(scratch, output->rows, 1);
    tensorSoftmax(probs, output);

    // 3. Backward pass
    // Inits output delta (softmax + cross-entropy "shortcut")
    Tensor *delta = tensorAlloc(scratch, output->rows, 1);
    for (int i = 0; i < output->rows; i++) {
        float target = (i == label) ? 1.0f : 0.0f;
        delta->data[i] = probs->data[i] - target;
    }

    for (int i = m->count - 1; i >= 0; i--) {
        Tensor *prevA = (i == 0) ? input : m->layers[i-1].a;
        
        // Accumulate weights and biases for present layer
        for (int r = 0; r < m->layers[i].w->rows; r++) {
            float d = delta->data[r];
            for (int c = 0; c < m->layers[i].w->cols; c++) {
                int idx = r * m->layers[i].w->cols + c;
                
                // Calc only the pure gradient here (reg. L2 is applied in update)
                float gradW = d * prevA->data[c];

                // Gradient clipping
                if (gradW > GRAD_CLIP) gradW = GRAD_CLIP;
                if (gradW < -GRAD_CLIP) gradW = -GRAD_CLIP;

                m->layers[i].gradW->data[idx] += gradW; // Accumulates
            }
            m->layers[i].gradB->data[r] += d; // Accumulates
        }

        // Propagate the error to the previous layer (ReLU Derivative)
        if (i > 0) {
            Tensor *upstreamDelta = tensorAlloc(scratch, m->layers[i-1].a->rows, 1);
            // W^T * delta
            for (int j = 0; j < m->layers[i].w->cols; j++) {
                float error = 0;
                for (int k = 0; k < m->layers[i].w->rows; k++) 
                    error += m->layers[i].w->data[k * m->layers[i].w->cols + j] * delta->data[k];
                upstreamDelta->data[j] = error;
            }
            Tensor *nextDelta = tensorAlloc(scratch, m->layers[i-1].a->rows, 1);
            tensorReLUDerivative(nextDelta, m->layers[i-1].z, upstreamDelta);
            delta = nextDelta;
        }
    }
}

void glueUpdateWeights(Model *m, float lr, int batchSize) {
    for (int i = 0; i < m->count; i++) {
        // Update weights
        for (int j = 0; j < m->layers[i].w->rows * m->layers[i].w->cols; j++) {
            // Calc the average of the gradient
            float grad = m->layers[i].gradW->data[j] / (float)batchSize;

            // L2 regularization: add lambda * weight to the gradient
            grad += LAMBDA * m->layers[i].w->data[j];

            // Subtract from weight
            m->layers[i].w->data[j] -= lr * grad;
            // Clean accumulator for next batch
            m->layers[i].gradW->data[j] = 0.0f;
        }

        // Update biases
        for (int j = 0; j < m->layers[i].b->rows; j++) {
            float avgGrad = m->layers[i].gradB->data[j] / batchSize;
            m->layers[i].b->data[j] -= lr * avgGrad;
            
            // Clean accumulator for next batch
            m->layers[i].gradB->data[j] = 0.0f;
        }
    }
}