# Compiler and Flags
# Detect OS if not set
OS ?= $(shell uname -s)

CC = gcc
CFLAGS = -Wall -Wextra -I./headers -I./headers/core -I./headers/cli -I./headers/dataset \
	-MMD -MP -I./headers/image -I./headers/utils -I./IO/external
CFLAGS += -O3 -march=native 
LIBS = -lm 

ifeq ($(OS),Darwin)
    OPENMP_CFLAGS = -Xpreprocessor -fopenmp
    OPENMP_LDFLAGS = -lomp
else
    OPENMP_CFLAGS = -fopenmp
    OPENMP_LDFLAGS = -lgomp
endif

CFLAGS += $(OPENMP_CFLAGS)
LIBS += $(OPENMP_LDFLAGS)

# Directories
SRCDIR = src
IODIR = IO
OBJDIR = obj

# Source files - recursively find all .c files in src/
SRC_ROOT = miniAI.c
SRC_CLI = $(wildcard $(SRCDIR)/cli/*.c)
SRC_CLI_COMMANDS = $(wildcard $(SRCDIR)/cli/commands/*.c)
SRC_CORE = $(wildcard $(SRCDIR)/core/*.c)
SRC_DATASET = $(wildcard $(SRCDIR)/dataset/*.c)
SRC_IMAGE = $(wildcard $(SRCDIR)/image/*.c)
SRC_UTILS = $(SRCDIR)/Utils.c
SRC_DATA = $(IODIR)/MemoryDatasets.c

# Object files with directory structure preserved
OBJ_ROOT = $(OBJDIR)/miniAI.o
OBJ_CLI = $(patsubst $(SRCDIR)/cli/%.c, $(OBJDIR)/cli/%.o, $(SRC_CLI))
OBJ_CLI_COMMANDS = $(patsubst $(SRCDIR)/cli/commands/%.c, $(OBJDIR)/cli/commands/%.o, $(SRC_CLI_COMMANDS))
OBJ_CORE = $(patsubst $(SRCDIR)/core/%.c, $(OBJDIR)/core/%.o, $(SRC_CORE))
OBJ_DATASET = $(patsubst $(SRCDIR)/dataset/%.c, $(OBJDIR)/dataset/%.o, $(SRC_DATASET))
OBJ_IMAGE = $(patsubst $(SRCDIR)/image/%.c, $(OBJDIR)/image/%.o, $(SRC_IMAGE))
OBJ_UTILS = $(OBJDIR)/Utils.o
OBJ_DATA = $(OBJDIR)/MemoryDatasets.o

OBJ_ALL = $(OBJ_ROOT) $(OBJ_CLI) $(OBJ_CLI_COMMANDS) $(OBJ_CORE) $(OBJ_DATASET) $(OBJ_IMAGE) $(OBJ_UTILS) $(OBJ_DATA)

# Executables
TARGET = miniAI

# Default target
all: $(OBJDIR) $(TARGET)

# Create obj directory structure
$(OBJDIR):
	mkdir -p $(OBJDIR)
	mkdir -p $(OBJDIR)/cli
	mkdir -p $(OBJDIR)/cli/commands
	mkdir -p $(OBJDIR)/core
	mkdir -p $(OBJDIR)/dataset
	mkdir -p $(OBJDIR)/image

# Compile root source file
$(OBJDIR)/miniAI.o: miniAI.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CLI source files
$(OBJDIR)/cli/%.o: $(SRCDIR)/cli/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CLI command files
$(OBJDIR)/cli/commands/%.o: $(SRCDIR)/cli/commands/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile core source files
$(OBJDIR)/core/%.o: $(SRCDIR)/core/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile dataset source files
$(OBJDIR)/dataset/%.o: $(SRCDIR)/dataset/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile image processing source files
$(OBJDIR)/image/%.o: $(SRCDIR)/image/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile utils
$(OBJDIR)/Utils.o: $(SRCDIR)/Utils.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile data files
$(OBJDIR)/MemoryDatasets.o: $(IODIR)/MemoryDatasets.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link main executable
$(TARGET): $(OBJ_ALL)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
	@echo ""
	@echo "========================================"
	@echo "  miniAI built successfully!"
	@echo "========================================"
	@echo ""

# Run shortcuts
run: $(TARGET)
	@./$(TARGET) help

train-static: $(TARGET)
	@./$(TARGET) train --dataset alpha --static

train-static-digits: $(TARGET)
	@./$(TARGET) train --dataset digits --static

train-png: $(TARGET)
	@./$(TARGET) train --data

train-png-digits: $(TARGET)
	@./$(TARGET) train --dataset digits --data

test: $(TARGET)
	@echo "Usage examples:"
	@echo "  make test-static    - Test static aloha model on static alpha dataset"
	@echo "  make test-png       - Test PNG alpha model on PNG alpha dataset"
	@echo "  make test-image IMG=test.png [MODEL=IO/models/digit_brain_png.bin]"

test-static: $(TARGET)
	@./$(TARGET) test --static

test-static-digits: $(TARGET)
	@./$(TARGET) test --static --dataset digits

test-png: $(TARGET)
	@./$(TARGET) test --data

test-png-digits: $(TARGET)
	@./$(TARGET) test --data --dataset digits

test-image: $(TARGET)
	@if [ -z "$(IMG)" ]; then \
		echo "Usage: make test-image IMG=<image.png>"; \
	else \
		./$(TARGET) test --image $(IMG); \
	fi

test-image-digits: $(TARGET)
	@if [ -z "$(IMG)" ]; then \
		echo "Usage: make test-image IMG=<image.png>"; \
	else \
		./$(TARGET) test --image $(IMG) --dataset digits; \
	fi

benchmark-static: $(TARGET)
	@./$(TARGET) benchmark --static --reps $(or $(REPS),3)

benchmark-static-digits: $(TARGET)
	@./$(TARGET) benchmark --dataset digits --static --reps $(or $(REPS),3)

benchmark-png: $(TARGET)
	@./$(TARGET) benchmark --data --reps $(or $(REPS),3)

benchmark-png-digits: $(TARGET)
	@./$(TARGET) benchmark --dataset digits --data --reps $(or $(REPS),3)

recognize: $(TARGET)
	@echo "Usage: make recognize-phrase IMG=phrase.png [MODEL=IO/models/alpha_brain_png.bin]"

recognize-phrase: $(TARGET)
	@./$(TARGET) recognize --model $(or $(MODEL),IO/models/alpha_brain_png.bin) --image $(IMG)

# Clean
clean:
	rm -rf $(OBJDIR)
	rm -f $(TARGET)

# Clean models only
clean-models:
	rm -f $(IODIR)/models/*.bin

# Clean configs only
clean-configs:
	rm -f $(IODIR)/configs/best_config*.txt

# Full clean (everything)
clean-all: clean clean-models clean-configs

# Full rebuild
rebuild: clean all

# Show structure
structure:
	@echo "Source structure:"
	@echo "  miniAI.c"
	@echo "  src/"
	@echo "    cli/          - CLI parsing and command dispatch"
	@echo "      commands/   - Command implementations"
	@echo "    core/         - Core neural network (Arena, Tensor, Model, etc)"
	@echo "    dataset/      - Dataset management and test utilities"
	@echo "    image/        - Image loading and preprocessing"
	@echo "    Utils.c       - General utilities"
	@echo "  IO/"
	@echo "    MemoryDatasets.c - Static in-memory datasets"
	@echo "    images/       - PNG datasets"
	@echo "    models/       - Trained models"
	@echo "    configs/      - Best hyperparameter configs (by benchmark)"

# Count lines of code
loc:
	@echo "Lines of code:"
	@wc -l miniAI.c $(SRCDIR)/*/*.c $(SRCDIR)/*/*/*.c $(IODIR)/*.c 2>/dev/null | tail -1

# Help
help:
	@echo "miniAI Makefile - Organized Structure"
	@echo ""
	@echo "Build targets:"
	@echo "  all              - Build miniAI executable"
	@echo "  clean            - Remove build files"
	@echo "  clean-models     - Remove trained models"
	@echo "  clean-configs    - Remove hyperparameter configs"
	@echo "  clean-all        - Remove everything (build + models + configs)"
	@echo "  rebuild          - Clean and rebuild"
	@echo ""
	@echo "Run shortcuts:"
	@echo "  make run         			- Show help"
	@echo "  make train-static       	- Train alpha (static)"
	@echo "  make train-static-digits	- Train digits (static)"
	@echo "  make train-png   			- Train alpha (PNG)"
	@echo "  make train-png-digits  	- Train digits (PNG)"
	@echo "  make test-static 			- Test alpha model (static)"
	@echo "  make test-static-digits	- Test digits model (static)"
	@echo "  make test-png    			- Test alpha model (PNG)"
	@echo "  make test-png-digits    	- Test PNG digits model (PNG)"
	@echo "  make benchmark-static   	- Benchmark alpha dataset (static)"
	@echo "  make benchmark-static-digits - Benchmark digits dataset (static)"
	@echo "  make benchmark-png			- Benchmark alpha dataset (PNG)"
	@echo "  make benchmark-png-digits	- Benchmark digits dataset (PNG)"
	@echo ""
	@echo "Info targets:"
	@echo "  make structure   - Show directory structure"
	@echo "  make loc         - Count lines of code"
	@echo "  make help        - Show help"
	@echo ""
	@echo "Direct usage:"
	@echo "  ./miniAI train --dataset digits --static"
	@echo "  ./miniAI train --dataset digits --data"
	@echo "  ./miniAI test --model IO/models/digit_brain_png.bin --image test.png"
	@echo "  ./miniAI benchmark --dataset digits --data --reps 5"
	@echo "  ./miniAI recognize --model IO/models/alpha_brain_png.bin --image phrase.png"
	@echo ""

.PHONY: all clean clean-models clean-configs clean-all rebuild run train \\
	train-digits train-png test test-static test-png test-image benchmark \\
	benchmark-png recognize recognize-phrase structure loc help

-include $(OBJ_ALL:.o=.d)