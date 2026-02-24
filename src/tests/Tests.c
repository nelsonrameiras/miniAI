#include "../../headers/tests/Tests.h"
#include "../../headers/core/Arena.h"
#include "../../headers/core/Tensor.h"
#include "../../headers/core/Grad.h"
#include "../../headers/core/Model.h"
#include "../../headers/core/Glue.h"
#include "../../headers/image/ImagePreprocess.h"
#include "../../headers/utils/Utils.h"
#include "../../AIHeader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

//  Minimal test framework

static int g_passed = 0;
static int g_failed = 0;

#define EXPECT(cond, msg) do { \
    if (cond) { \
        printf("  PASS  %s\n", msg); \
        g_passed++; \
    } else { \
        printf("  FAIL  %s  [%s:%d]\n", msg, __FILE__, __LINE__); \
        g_failed++; \
    } \
} while(0)

static void suiteBegin(const char *name) {
    printf("\n--- %s ---\n", name);
}

static int feq(float a, float b)             { return fabsf(a - b) < 1e-5f; }
static int feqr(float a, float b, float tol) { return fabsf(a - b) < tol; }
static int fisvalid(float x)                 { return !isnan(x) && !isinf(x); }

//  1. Arena

int testArena(void) {
    suiteBegin("Arena");

    Arena *a = arenaInit(1024);
    EXPECT(a != NULL,                         "arenaInit returns non-NULL");
    EXPECT(a->used == 0,                      "initial used == 0");
    EXPECT(a->capacity == 1024,               "capacity matches requested size");
    EXPECT(arenaRemainingCapacity(a) == 1024, "remaining == capacity initially");

    void *p1 = arenaAlloc(a, 16);
    EXPECT(p1 != NULL,                        "arenaAlloc 16 bytes succeeds");
    EXPECT(a->used == 16,                     "used == 16 after 16-byte alloc");

    // 8-byte alignment: 7 bytes must consume 8
    void *p2 = arenaAlloc(a, 7);
    EXPECT(p2 != NULL,                        "arenaAlloc 7 bytes succeeds");
    EXPECT(a->used == 24,                     "used == 24 after 7-byte alloc (aligned to 8)");
    EXPECT((char*)p2 >= (char*)p1 + 16,       "allocations do not overlap");

    // Zero-initialisation
    unsigned char *bytes = (unsigned char*)arenaAlloc(a, 8);
    int allZero = 1;
    for (int i = 0; i < 8; i++) if (bytes[i] != 0) allZero = 0;
    EXPECT(allZero,                           "arenaAlloc returns zeroed memory");

    // Exact capacity fill
    arenaReset(a);
    void *full = arenaAlloc(a, 1024);
    EXPECT(full != NULL,                      "arenaAlloc exactly to capacity succeeds");
    EXPECT(arenaRemainingCapacity(a) == 0,    "remaining == 0 after full alloc");

    // Over capacity must fail
    arenaReset(a);
    arenaAlloc(a, 1017);   // aligns to 1024
    void *over = arenaAlloc(a, 8);
    EXPECT(over == NULL,                      "arenaAlloc beyond capacity returns NULL");

    // Reset restores availability
    arenaReset(a);
    EXPECT(a->used == 0,                      "arenaReset sets used == 0");
    EXPECT(arenaRemainingCapacity(a) == 1024, "remaining == capacity after reset");
    void *p3 = arenaAlloc(a, 32);
    EXPECT(p3 != NULL,                        "arenaAlloc after reset succeeds");

    // Two independent arenas do not interfere
    Arena *b = arenaInit(512);
    void *pb = arenaAlloc(b, 64);
    void *pa = arenaAlloc(a, 64);
    EXPECT(pb != NULL && pa != NULL,          "two independent arenas both allocate");
    EXPECT((char*)pb < (char*)a->buffer ||
           (char*)pb >= (char*)a->buffer + a->capacity,
                                              "arena B pointer outside arena A buffer");
    arenaFree(b);
    arenaFree(a);
    EXPECT(1,                                 "arenaFree does not crash");

    return g_failed;
}

// ============================================================
//  2. Tensor
// ============================================================

int testTensor(void) {
    suiteBegin("Tensor");

    Arena *a = arenaInit(256 * 1024);

    // --- Allocation ---
    Tensor *t = tensorAlloc(a, 3, 2);
    EXPECT(t != NULL,                         "tensorAlloc returns non-NULL");
    EXPECT(t->rows == 3 && t->cols == 2,      "rows/cols set correctly");
    int allZero = 1;
    for (int i = 0; i < 6; i++) if (t->data[i] != 0.0f) allZero = 0;
    EXPECT(allZero,                           "tensorAlloc data is zero-initialised");

    EXPECT(tensorAlloc(a, 0, 5)  == NULL,     "tensorAlloc: zero rows returns NULL");
    EXPECT(tensorAlloc(a, 5, 0)  == NULL,     "tensorAlloc: zero cols returns NULL");
    EXPECT(tensorAlloc(a, -1, 4) == NULL,     "tensorAlloc: negative rows returns NULL");

    // --- tensorDot ---
    // [1 2] * [5] = [17]
    // [3 4]   [6]   [39]
    Tensor *ma = tensorAlloc(a, 2, 2);
    Tensor *mb = tensorAlloc(a, 2, 1);
    Tensor *mc = tensorAlloc(a, 2, 1);
    ma->data[0]=1; ma->data[1]=2; ma->data[2]=3; ma->data[3]=4;
    mb->data[0]=5; mb->data[1]=6;
    tensorDot(mc, ma, mb);
    EXPECT(feq(mc->data[0], 17.0f),           "tensorDot: result[0] == 17");
    EXPECT(feq(mc->data[1], 39.0f),           "tensorDot: result[1] == 39");

    // Identity matrix: I * v = v
    Tensor *eye = tensorAlloc(a, 3, 3);
    Tensor *vec = tensorAlloc(a, 3, 1);
    Tensor *res = tensorAlloc(a, 3, 1);
    eye->data[0]=1; eye->data[4]=1; eye->data[8]=1;
    vec->data[0]=7; vec->data[1]=-3; vec->data[2]=2;
    tensorDot(res, eye, vec);
    EXPECT(feq(res->data[0], 7.0f) &&
           feq(res->data[1], -3.0f) &&
           feq(res->data[2], 2.0f),           "tensorDot: I * v == v");

    // --- tensorAdd ---
    Tensor *aa = tensorAlloc(a, 2, 1);
    Tensor *ab = tensorAlloc(a, 2, 1);
    Tensor *ac = tensorAlloc(a, 2, 1);
    aa->data[0]=1.5f; aa->data[1]=-2.0f;
    ab->data[0]=0.5f; ab->data[1]=3.0f;
    tensorAdd(ac, aa, ab);
    EXPECT(feq(ac->data[0], 2.0f),            "tensorAdd: result[0] == 2.0");
    EXPECT(feq(ac->data[1], 1.0f),            "tensorAdd: result[1] == 1.0");

    // Commutativity
    Tensor *ba_res = tensorAlloc(a, 2, 1);
    tensorAdd(ba_res, ab, aa);
    EXPECT(feq(ba_res->data[0], ac->data[0]) &&
           feq(ba_res->data[1], ac->data[1]), "tensorAdd: commutative");

    // --- tensorSigmoid ---
    Tensor *si = tensorAlloc(a, 5, 1);
    Tensor *so = tensorAlloc(a, 5, 1);
    si->data[0]=0.0f;    // 0.5
    si->data[1]=100.0f;  // ~1
    si->data[2]=-100.0f; // ~0
    si->data[3]=1.0f;    // ~0.731
    si->data[4]=-1.0f;   // ~0.269
    tensorSigmoid(so, si);
    EXPECT(feq(so->data[0], 0.5f),            "tensorSigmoid(0) == 0.5");
    EXPECT(feqr(so->data[1], 1.0f, 1e-4f),   "tensorSigmoid(+large) ~= 1");
    EXPECT(feqr(so->data[2], 0.0f, 1e-4f),   "tensorSigmoid(-large) ~= 0");
    // f(x) + f(-x) == 1 (symmetry)
    EXPECT(feqr(so->data[3] + so->data[4], 1.0f, 1e-5f),
                                              "tensorSigmoid: f(x) + f(-x) == 1");
    // Moderate inputs (±1) must be strictly in (0, 1)
    EXPECT(so->data[3] > 0.0f && so->data[3] < 1.0f &&
           so->data[4] > 0.0f && so->data[4] < 1.0f,
                                              "tensorSigmoid: moderate inputs strictly in (0, 1)");
    // Monotonically increasing: sigmoid(-1) < sigmoid(0) < sigmoid(1)
    EXPECT(so->data[4] < so->data[0] && so->data[0] < so->data[3],
                                              "tensorSigmoid: monotonically increasing");

    // --- tensorSoftmax ---
    Tensor *smin = tensorAlloc(a, 3, 1);
    Tensor *smout = tensorAlloc(a, 3, 1);
    smin->data[0]=0.0f; smin->data[1]=1.0f; smin->data[2]=2.0f;
    tensorSoftmax(smout, smin);
    float smsum = smout->data[0] + smout->data[1] + smout->data[2];
    EXPECT(feqr(smsum, 1.0f, 1e-5f),         "tensorSoftmax: sums to 1.0");
    EXPECT(smout->data[0] > 0 &&
           smout->data[1] > 0 &&
           smout->data[2] > 0,               "tensorSoftmax: all outputs positive");
    EXPECT(smout->data[2] > smout->data[1] &&
           smout->data[1] > smout->data[0],  "tensorSoftmax: preserves ordering");

    // Uniform input -> uniform output (1/n each)
    Tensor *uni  = tensorAlloc(a, 4, 1);
    Tensor *unio = tensorAlloc(a, 4, 1);
    for (int i = 0; i < 4; i++) uni->data[i] = 1.0f;
    tensorSoftmax(unio, uni);
    EXPECT(feqr(unio->data[0], 0.25f, 1e-5f) &&
           feqr(unio->data[3], 0.25f, 1e-5f),"tensorSoftmax: uniform input -> 1/n each");

    // Numerical stability with large values
    Tensor *largeIn  = tensorAlloc(a, 3, 1);
    Tensor *largeOut = tensorAlloc(a, 3, 1);
    largeIn->data[0]=1000.0f; largeIn->data[1]=1001.0f; largeIn->data[2]=1002.0f;
    tensorSoftmax(largeOut, largeIn);
    float lsum = largeOut->data[0] + largeOut->data[1] + largeOut->data[2];
    EXPECT(fisvalid(lsum),                    "tensorSoftmax: no NaN/Inf with large inputs");
    EXPECT(feqr(lsum, 1.0f, 1e-5f),          "tensorSoftmax: sum == 1.0 with large inputs");

    // --- tensorReLU ---
    Tensor *ri = tensorAlloc(a, 4, 1);
    Tensor *ro = tensorAlloc(a, 4, 1);
    ri->data[0]=-2.0f; ri->data[1]=0.0f; ri->data[2]=1.0f; ri->data[3]=-0.001f;
    tensorReLU(ro, ri);
    EXPECT(feq(ro->data[0], 0.0f),            "tensorReLU(-2)     == 0");
    EXPECT(feq(ro->data[1], 0.0f),            "tensorReLU(0)      == 0");
    EXPECT(feq(ro->data[2], 1.0f),            "tensorReLU(1)      == 1");
    EXPECT(feq(ro->data[3], 0.0f),            "tensorReLU(-0.001) == 0");

    // --- tensorFillXavier ---
    // All values must be in [-scale, scale] where scale = sqrt(2/fanIn)
    // Mean must be close to 0 (statistical)
    Tensor *xav = tensorAlloc(a, 100, 100);
    int fanIn = 64;
    tensorFillXavier(xav, fanIn);
    float scale = sqrtf(2.0f / (float)fanIn);
    float xmean = 0.0f;
    int outOfRange = 0;
    int total = xav->rows * xav->cols;
    for (int i = 0; i < total; i++) {
        xmean += xav->data[i];
        if (xav->data[i] < -scale || xav->data[i] > scale) outOfRange++;
    }
    xmean /= total;
    EXPECT(fabsf(xmean) < 0.05f,             "tensorFillXavier: mean close to 0");
    EXPECT(outOfRange == 0,                  "tensorFillXavier: all values in [-scale, scale]");

    // Values must not all be identical (not degenerate)
    Tensor *xav2 = tensorAlloc(a, 4, 4);
    tensorFillXavier(xav2, 4);
    int hasVariance = 0;
    for (int i = 1; i < 16; i++)
        if (!feq(xav2->data[i], xav2->data[0])) { hasVariance = 1; break; }
    EXPECT(hasVariance,                       "tensorFillXavier: produces varied values");

    arenaFree(a);
    return g_failed;
}

// ============================================================
//  3. Grad
// ============================================================

int testGrad(void) {
    suiteBegin("Grad");

    Arena *a = arenaInit(8 * 1024);

    // --- sigmoidDerivative ---
    EXPECT(feq(sigmoidDerivative(0.0f), 0.25f),
                                              "sigmoidDerivative(0) == 0.25");
    EXPECT(sigmoidDerivative(100.0f) < 1e-4f,
                                              "sigmoidDerivative(+large) ~= 0");
    EXPECT(sigmoidDerivative(-100.0f) < 1e-4f,
                                              "sigmoidDerivative(-large) ~= 0");
    EXPECT(feq(sigmoidDerivative(2.0f), sigmoidDerivative(-2.0f)),
                                              "sigmoidDerivative: f'(x) == f'(-x)");
    EXPECT(sigmoidDerivative(-5.0f) >= 0.0f &&
           sigmoidDerivative(0.0f)  >= 0.0f &&
           sigmoidDerivative(5.0f)  >= 0.0f, "sigmoidDerivative: always >= 0");
    EXPECT(sigmoidDerivative(0.0f) > sigmoidDerivative(1.0f),
                                              "sigmoidDerivative: peak at x=0");

    // --- tensorSigmoidPrime ---
    Tensor *sp_in  = tensorAlloc(a, 3, 1);
    Tensor *sp_out = tensorAlloc(a, 3, 1);
    sp_in->data[0]=0.0f; sp_in->data[1]=1.0f; sp_in->data[2]=-1.0f;
    tensorSigmoidPrime(sp_out, sp_in);
    EXPECT(feq(sp_out->data[0], 0.25f),       "tensorSigmoidPrime(0) == 0.25");
    EXPECT(feq(sp_out->data[1], sigmoidDerivative(1.0f)),
                                              "tensorSigmoidPrime(1) matches scalar");
    EXPECT(feq(sp_out->data[2], sigmoidDerivative(-1.0f)),
                                              "tensorSigmoidPrime(-1) matches scalar");

    // --- tensorReLUDerivative ---
    Tensor *z    = tensorAlloc(a, 5, 1);
    Tensor *up   = tensorAlloc(a, 5, 1);
    Tensor *dout = tensorAlloc(a, 5, 1);
    z->data[0]=-2.0f; z->data[1]=-0.001f; z->data[2]=0.0f;
    z->data[3]=0.001f; z->data[4]=3.0f;
    for (int i = 0; i < 5; i++) up->data[i] = 5.0f;
    tensorReLUDerivative(dout, z, up);
    EXPECT(feq(dout->data[0], 0.0f),          "ReLUDerivative: negative z => 0");
    EXPECT(feq(dout->data[1], 0.0f),          "ReLUDerivative: small negative z => 0");
    EXPECT(feq(dout->data[2], 0.0f),          "ReLUDerivative: z==0 => 0");
    EXPECT(feq(dout->data[3], 5.0f),          "ReLUDerivative: small positive z => upstream");
    EXPECT(feq(dout->data[4], 5.0f),          "ReLUDerivative: large positive z => upstream");

    // Scales correctly with upstream magnitude
    Tensor *up2  = tensorAlloc(a, 2, 1);
    Tensor *z2   = tensorAlloc(a, 2, 1);
    Tensor *out2 = tensorAlloc(a, 2, 1);
    z2->data[0]=1.0f;  z2->data[1]=1.0f;
    up2->data[0]=2.0f; up2->data[1]=7.0f;
    tensorReLUDerivative(out2, z2, up2);
    EXPECT(feq(out2->data[0], 2.0f) &&
           feq(out2->data[1], 7.0f),          "ReLUDerivative: scales with upstream gradient");

    arenaFree(a);
    return g_failed;
}

// ============================================================
//  4. Shuffle
// ============================================================

int testShuffle(void) {
    suiteBegin("Shuffle");

    int n = 10;
    int arr[10];
    for (int i = 0; i < n; i++) arr[i] = i;
    shuffle(arr, n);

    int seen[10] = {0};
    for (int i = 0; i < n; i++) {
        EXPECT(arr[i] >= 0 && arr[i] < n,    "shuffle: element in valid range");
        seen[arr[i]]++;
    }
    int allSeen = 1;
    for (int i = 0; i < n; i++) if (seen[i] != 1) allSeen = 0;
    EXPECT(allSeen,                           "shuffle: all original elements present exactly once");

    // Edge cases
    int single = 42;
    shuffle(&single, 1);
    EXPECT(single == 42,                      "shuffle: length-1 unchanged");
    shuffle(arr, 0);
    EXPECT(1,                                 "shuffle: length-0 does not crash");

    // Statistical: 20-element shuffle almost certainly not identity (1/20! chance)
    int big[20];
    for (int i = 0; i < 20; i++) big[i] = i;
    shuffle(big, 20);
    int isIdentity = 1;
    for (int i = 0; i < 20; i++) if (big[i] != i) { isIdentity = 0; break; }
    EXPECT(!isIdentity,                       "shuffle: 20-element result is not identity permutation");

    return g_failed;
}

// ============================================================
//  5. Model (structure, save/load round-trip)
// ============================================================

int testModel(void) {
    suiteBegin("Model");

    Arena *perm = arenaInit(4 * 1024 * 1024);
    int dims[] = {4, 8, 3};

    // --- Structure ---
    Model *m = modelCreate(perm, dims, 3);
    EXPECT(m != NULL,                         "modelCreate returns non-NULL");
    EXPECT(m->count == 2,                     "modelCreate: count == num_dims - 1");

    EXPECT(m->layers[0].w->rows == 8 &&
           m->layers[0].w->cols == 4,         "layer 0 weights: 8x4");
    EXPECT(m->layers[0].b->rows == 8 &&
           m->layers[0].b->cols == 1,         "layer 0 bias: 8x1");
    EXPECT(m->layers[1].w->rows == 3 &&
           m->layers[1].w->cols == 8,         "layer 1 weights: 3x8");
    EXPECT(m->layers[1].b->rows == 3 &&
           m->layers[1].b->cols == 1,         "layer 1 bias: 3x1");

    // Gradient accumulators zero-initialised
    int gradsZero = 1;
    for (int l = 0; l < m->count; l++) {
        for (int i = 0; i < m->layers[l].gradW->rows * m->layers[l].gradW->cols; i++)
            if (m->layers[l].gradW->data[i] != 0.0f) gradsZero = 0;
        for (int i = 0; i < m->layers[l].gradB->rows; i++)
            if (m->layers[l].gradB->data[i] != 0.0f) gradsZero = 0;
    }
    EXPECT(gradsZero,                         "modelCreate: gradient accumulators zero-initialised");

    // Weights Xavier-initialised (not all zero)
    int weightsNonZero = 0;
    for (int i = 0; i < m->layers[0].w->rows * m->layers[0].w->cols; i++)
        if (m->layers[0].w->data[i] != 0.0f) { weightsNonZero = 1; break; }
    EXPECT(weightsNonZero,                    "modelCreate: weights Xavier-initialised (not all zero)");

    // --- Save / Load round-trip ---
    const char *tmpFile = "/tmp/miniAI_test_model.bin";
    EXPECT(modelSave(m, tmpFile) == 0,        "modelSave: returns 0 on success");

    Arena *perm2 = arenaInit(4 * 1024 * 1024);
    Model *m2    = modelCreate(perm2, dims, 3);
    EXPECT(m2 != NULL,                        "modelCreate for load target succeeds");
    EXPECT(modelLoad(m2, tmpFile) == 0,       "modelLoad: returns 0 on success");

    // Every weight and bias must be bit-identical
    int weightsMatch = 1, biasesMatch = 1;
    for (int l = 0; l < m->count; l++) {
        int wSize = m->layers[l].w->rows * m->layers[l].w->cols;
        for (int i = 0; i < wSize; i++)
            if (m->layers[l].w->data[i] != m2->layers[l].w->data[i]) weightsMatch = 0;
        for (int i = 0; i < m->layers[l].b->rows; i++)
            if (m->layers[l].b->data[i] != m2->layers[l].b->data[i]) biasesMatch = 0;
    }
    EXPECT(weightsMatch,                      "modelSave/Load: all weights bit-identical after round-trip");
    EXPECT(biasesMatch,                       "modelSave/Load: all biases bit-identical after round-trip");

    // Load into wrong architecture must fail
    int dimsMismatch[] = {4, 16, 3};
    Arena *perm3 = arenaInit(4 * 1024 * 1024);
    Model *m3    = modelCreate(perm3, dimsMismatch, 3);
    EXPECT(modelLoad(m3, tmpFile) != 0,       "modelLoad: error on architecture mismatch");

    // Load from nonexistent file must fail
    EXPECT(modelLoad(m2, "/tmp/miniAI_no_such_file.bin") != 0,
                                              "modelLoad: error on missing file");

    remove(tmpFile);
    arenaFree(perm3);
    arenaFree(perm2);
    arenaFree(perm);
    return g_failed;
}

// ============================================================
//  6. Glue (forward pass shape, predict, loss, backprop direction)
// ============================================================

int testGlue(void) {
    suiteBegin("Glue");

    int dims[]    = {4, 8, 3};
    int inputSize = 4, outputSize = 3;

    Arena *perm    = arenaInit(2 * 1024 * 1024);
    Arena *scratch = arenaInit(1 * 1024 * 1024);

    TrainingConfig savedCfg = g_trainConfig;

    Model *m = modelCreate(perm, dims, 3);
    EXPECT(m != NULL,                         "glue: modelCreate succeeds");

    // --- glueForward output shape and validity ---
    // Fresh alloc before each forward pass - scratch arena is reset between calls
    Tensor *input = tensorAlloc(scratch, inputSize, 1);
    for (int i = 0; i < inputSize; i++) input->data[i] = 0.5f;

    Tensor *output = glueForward(m, input, scratch);
    EXPECT(output != NULL,                    "glueForward: returns non-NULL");
    EXPECT(output->rows == outputSize,        "glueForward: output rows == outputSize");
    EXPECT(output->cols == 1,                 "glueForward: output is column vector");

    int outputFinite = 1;
    for (int i = 0; i < outputSize; i++)
        if (!fisvalid(output->data[i])) outputFinite = 0;
    EXPECT(outputFinite,                      "glueForward: all output values finite");

    // --- gluePredict valid class ---
    // Must re-allocate input after every arenaReset: the arena resets its allocation pointer, so the old input pointer is no longer valid.
    arenaReset(scratch);
    float conf = -1.0f;
    input = tensorAlloc(scratch, inputSize, 1);
    for (int i = 0; i < inputSize; i++) input->data[i] = 0.5f;
    int pred = gluePredict(m, input, scratch, &conf);
    EXPECT(pred >= 0 && pred < outputSize,    "gluePredict: returns valid class index");
    EXPECT(conf > 0.0f && conf <= 1.0f,       "gluePredict: confidence in (0, 1]");

    arenaReset(scratch);
    input = tensorAlloc(scratch, inputSize, 1);
    for (int i = 0; i < inputSize; i++) input->data[i] = 0.5f;
    int pred2 = gluePredict(m, input, scratch, NULL);
    EXPECT(pred2 >= 0 && pred2 < outputSize,  "gluePredict: works with NULL confidence pointer");


    // --- glueComputeLoss positive and finite ---
    arenaReset(scratch);
    input = tensorAlloc(scratch, inputSize, 1);
    for (int i = 0; i < inputSize; i++) input->data[i] = 0.5f;
    output = glueForward(m, input, scratch);
    float loss = glueComputeLoss(output, 0, scratch);
    EXPECT(loss > 0.0f,                       "glueComputeLoss: loss > 0");
    EXPECT(fisvalid(loss),                    "glueComputeLoss: loss is finite");

    // Perfect prediction should give lower loss than random
    // Run one gradient step toward label 0, loss should decrease
    float rawInput[4] = {1.0f, 0.0f, 1.0f, 0.0f};
    int label = 0;

    arenaReset(scratch);
    Tensor *inp0 = tensorAlloc(scratch, inputSize, 1);
    memcpy(inp0->data, rawInput, inputSize * sizeof(float));
    float lossBefore = glueComputeLoss(glueForward(m, inp0, scratch), label, scratch);

    // Several gradient steps toward label
    for (int s = 0; s < 50; s++) {
        arenaReset(scratch);
        glueAccumulateGradients(m, rawInput, label, 0.0f, scratch);
        glueUpdateWeights(m, 0.05f, 1);
    }

    arenaReset(scratch);
    Tensor *inp1 = tensorAlloc(scratch, inputSize, 1);
    memcpy(inp1->data, rawInput, inputSize * sizeof(float));
    float lossAfter = glueComputeLoss(glueForward(m, inp1, scratch), label, scratch);

    EXPECT(lossAfter < lossBefore,            "backprop: loss decreases after gradient steps");

    // Prediction should converge to the trained label
    arenaReset(scratch);
    Tensor *finalInp = tensorAlloc(scratch, inputSize, 1);
    memcpy(finalInp->data, rawInput, inputSize * sizeof(float));
    EXPECT(gluePredict(m, finalInp, scratch, NULL) == label,
                                              "backprop: predict converges to correct label");

    // --- Loss ordering: correct label < wrong label ---
    // After training on label 0, the loss for label 0 must be lower than for label 1
    arenaReset(scratch);
    Tensor *lossInp = tensorAlloc(scratch, inputSize, 1);
    memcpy(lossInp->data, rawInput, inputSize * sizeof(float));
    Tensor *lossOut = glueForward(m, lossInp, scratch);
    float lossCorrect = glueComputeLoss(lossOut, 0, scratch);
    float lossWrong   = glueComputeLoss(lossOut, 1, scratch);
    EXPECT(lossCorrect < lossWrong,           "glueComputeLoss: correct label < wrong label after training");

    // --- Numeric stability: loss must never be NaN or Inf ---
    // Even with a freshly initialised (random weights) model
    Arena *perm2   = arenaInit(1 * 1024 * 1024);
    Arena *scratch2 = arenaInit(512 * 1024);
    Model *m2 = modelCreate(perm2, dims, 3);
    int allLossValid = 1;
    for (int trial = 0; trial < 10; trial++) {
        arenaReset(scratch2);
        Tensor *trialInp = tensorAlloc(scratch2, inputSize, 1);
        for (int i = 0; i < inputSize; i++) trialInp->data[i] = (float)rand() / RAND_MAX;
        Tensor *trialOut = glueForward(m2, trialInp, scratch2);
        float trialLoss  = glueComputeLoss(trialOut, rand() % outputSize, scratch2);
        if (!fisvalid(trialLoss) || trialLoss <= 0.0f) allLossValid = 0;
    }
    EXPECT(allLossValid,                      "glueComputeLoss: always finite and positive (10 random trials)");
    arenaFree(scratch2);
    arenaFree(perm2);

    g_trainConfig = savedCfg;
    arenaFree(scratch);
    arenaFree(perm);
    return g_failed;
}

// ============================================================
//  7. Image Preprocessing (pure functions)
// ============================================================

int testImagePreprocess(void) {
    suiteBegin("ImagePreprocess");

    // --- rgbToGray ---
    EXPECT(rgbToGray(255, 255, 255) == 255,         "rgbToGray: white -> 255");
    EXPECT(rgbToGray(0, 0, 0) == 0,                 "rgbToGray: black -> 0");

    // ITU-R 601 coefficients: 0.299 R + 0.587 G + 0.114 B
    uint8_t gR = rgbToGray(255, 0, 0);
    uint8_t gG = rgbToGray(0, 255, 0);
    uint8_t gB = rgbToGray(0, 0, 255);
    EXPECT(gR >= 75 && gR <= 77,                    "rgbToGray: pure red ~= 76");
    EXPECT(gG >= 148 && gG <= 150,                  "rgbToGray: pure green ~= 149");
    EXPECT(gB >= 28 && gB <= 30,                    "rgbToGray: pure blue ~= 29");
    // Green channel contributes most (highest luminance coefficient)
    EXPECT(gG > gR && gR > gB,                      "rgbToGray: green > red > blue (per coefficients)");
    // Monotonic with uniform brightness
    EXPECT(rgbToGray(100,100,100) <= rgbToGray(200,200,200),
                                                    "rgbToGray: monotonic with brightness");

    // --- calculateOtsuThreshold ---
    // Perfectly bimodal [0, 255]: threshold must be between the two clusters
    int totalPixels = 512;
    uint8_t *bimodal = (uint8_t*)malloc(totalPixels);
    for (int i = 0; i < totalPixels / 2; i++) bimodal[i] = 0;
    for (int i = totalPixels / 2; i < totalPixels; i++) bimodal[i] = 255;
    uint8_t t1 = calculateOtsuThreshold(bimodal, totalPixels);
    EXPECT(t1 > 0 && t1 < 255,                      "Otsu: [0,255] bimodal -> threshold between clusters");
    free(bimodal);

    // Uniform image: all same value — must not crash, threshold within [0,254]
    uint8_t *uniform = (uint8_t*)malloc(256);
    for (int i = 0; i < 256; i++) uniform[i] = 128;
    uint8_t t2 = calculateOtsuThreshold(uniform, 256);
    EXPECT((int)t2 < 255,                               "Otsu: uniform image -> threshold < 255, no crash");
    free(uniform);

    // Two well-separated clusters [50, 200]: threshold must fall between them
    uint8_t *known = (uint8_t*)malloc(200);
    for (int i = 0; i < 100; i++) known[i] = 50;
    for (int i = 100; i < 200; i++) known[i] = 200;
    uint8_t t3 = calculateOtsuThreshold(known, 200);
    EXPECT(t3 > 50 && t3 < 200,                     "Otsu: clusters at 50/200 -> threshold between them");
    free(known);

    // Binary image [0, 255]: the +1 correction in the implementation
    // should push threshold above 0 so foreground (dark) pixels are classified correctly
    uint8_t *binary = (uint8_t*)malloc(100);
    for (int i = 0; i < 50; i++) binary[i] = 0;
    for (int i = 50; i < 100; i++) binary[i] = 255;
    uint8_t t4 = calculateOtsuThreshold(binary, 100);
    EXPECT(t4 > 0,                                  "Otsu: binary image threshold corrected above 0");
    EXPECT((int)t4 < 255,                           "Otsu: binary image threshold below max value");
    free(binary);

    return g_failed;
}

// ============================================================
//  Entry point
// ============================================================

int runAllTests(void) {
    g_passed = 0;
    g_failed = 0;

    testArena();
    testTensor();
    testGrad();
    testShuffle();
    testModel();
    testGlue();
    testImagePreprocess();

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", g_passed, g_failed);
    printf("========================================\n\n");

    return g_failed;
}

int main(void) {
    int failures = runAllTests();
    return failures > 0 ? 1 : 0;
}