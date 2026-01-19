# Compiler and Flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -I./headers -I./IO/external
LIBS = -lm

# Directories
SRCDIR = src
TESTDIR = src/tests
IODIR = IO
HEADDIR = headers
OBJDIR = obj

# Source files
SRC_CORE = $(wildcard $(SRCDIR)/*.c)
SRC_IO = $(IODIR)/ImageLoader.c $(IODIR)/ImagePreprocess.c
SRC_TESTS = $(TESTDIR)/testDriver.c $(TESTDIR)/testDriverImage.c $(TESTDIR)/testDriverSimple.c

# Object files
OBJ_CORE = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRC_CORE))
OBJ_IO = $(OBJDIR)/ImageLoader.o $(OBJDIR)/ImagePreprocess.o
OBJ_ALL = $(OBJ_CORE) $(OBJ_IO)

# Executables
TARGET_DRIVER = testDriver
TARGET_DRIVER_IMAGE = testDriverPNG
TARGET_DRIVER_SIMPLE = testDriverSimple

# Default target
all: $(OBJDIR) $(TARGET_DRIVER) $(TARGET_DRIVER_IMAGE) $(TARGET_DRIVER_SIMPLE)

# Create obj directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compile core source files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile IO source files
$(OBJDIR)/ImageLoader.o: $(IODIR)/ImageLoader.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/ImagePreprocess.o: $(IODIR)/ImagePreprocess.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile test driver object files
$(OBJDIR)/testDriver.o: $(TESTDIR)/testDriver.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/testDriverImage.o: $(TESTDIR)/testDriverImage.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/testDriverSimple.o: $(TESTDIR)/testDriverSimple.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link main executable (demo = testDriver + PNG support)
$(TARGET_DRIVER): $(OBJ_ALL) $(OBJDIR)/testDriver.o
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

# Link testDriverImage
$(TARGET_DRIVER_IMAGE): $(OBJ_ALL) $(OBJDIR)/testDriverImage.o
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

# Link testDriverSimple
$(TARGET_DRIVER_SIMPLE): $(OBJ_ALL) $(OBJDIR)/testDriverSimple.o
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

# Run shortcuts
run: $(TARGET_DRIVER)
	@./$(TARGET_DRIVER)

run-image: $(TARGET_DRIVER_IMAGE)
	@./$(TARGET_DRIVER_IMAGE)

run-simple: $(TARGET_DRIVER_SIMPLE)
	@./$(TARGET_DRIVER_SIMPLE)

# Test with PNG
test-png: $(TARGET_DRIVER)
	@./$(TARGET_DRIVER) --test-png $(PNG)

# Clean
clean:
	rm -rf $(OBJDIR)
	rm -f $(TARGET_DRIVER) $(TARGET_DRIVER_IMAGE) $(TARGET_DRIVER_SIMPLE)

# Clean models only
clean-models:
	rm -f $(IODIR)/models/*.bin
	rm -f $(IODIR)/confs/best_config*.txt

# Full rebuild
rebuild: clean all

# Help
help:
	@echo "miniAI Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build demo (main executable)"
	@echo "  all-targets      - Build all executables"
	@echo "  clean            - Remove all build files"
	@echo "  clean-models     - Remove trained models only"
	@echo "  rebuild          - Clean and rebuild everything"
	@echo ""
	@echo "Run shortcuts:"
	@echo "  make run         - Run demo"
	@echo "  make run-image   - Run testDriverImage"
	@echo "  make run-simple  - Run testDriverSimple"
	@echo "  make test-png PNG=image.png - Test with specific PNG"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build demo"
	@echo "  make run                # Train model"
	@echo "  make test-png PNG=IO/pngAlphaChars/065_A.png"
	@echo "  make clean-models       # Delete old models before retraining"

.PHONY: all all-targets clean clean-models rebuild run run-image run-simple test-png help