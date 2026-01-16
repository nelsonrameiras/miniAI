# Compiler and Flags
CC = gcc
CFLAGS = -Wall -Wextra -O3 -Iheaders
LIBS = -lm

# Directories
SRCDIR = src
HEADDIR = headers
OBJDIR = obj

SRC_ROOT = $(wildcard *.c)
SRC_LIBS = $(wildcard $(SRCDIR)/*.c)

OBJ_ROOT = $(patsubst %.c, $(OBJDIR)/%.o, $(SRC_ROOT))
OBJ_LIBS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRC_LIBS))
OBJECTS = $(OBJ_ROOT) $(OBJ_LIBS)

TARGET = mini_ai_demo

# Default Rule
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

$(OBJDIR)/%.o: %.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

ifeq (run,$(firstword $(MAKECMDGOALS)))
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

# Run the demo
run: all
	@./$(TARGET) $(RUN_ARGS)

# Clean build files
clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean run