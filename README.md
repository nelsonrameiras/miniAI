# mini_ai — Academic Mini AI in C


This project implements a minimal, academically-oriented AI system in C. Key features:


* Custom arena allocator
* 2D Tensor abstraction (row-major) built on top of the arena
* Explicit gradient math for logistic regression
* Model + glue layer demonstrating training loop and inference
* Simple test drivers and demo


## Building


Requires a POSIX C compiler (gcc recommended). From the project root:


```bash
make
./bin/testDriver