#ifndef TESTS_H
#define TESTS_H

// Run all unit test suites. Returns number of failed tests (0 = all passed).
int runAllTests(void);

// Individual suites (can be called independently)
int testArena(void);
int testTensor(void);
int testGrad(void);
int testShuffle(void);
int testModel(void);
int testGlue(void);
int testImagePreprocess(void);

#endif // TESTS_H