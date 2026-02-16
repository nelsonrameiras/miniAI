# miniAI Docker Image
# Multi-stage build for minimal final image

# Stage 1: Build
FROM gcc:latest AS builder

WORKDIR /build

# Copy source files
COPY . .

# Build miniAI with optimizations
RUN make clean && \
    make CFLAGS="-Wall -Wextra -O3 -DNDEBUG -fopenmp" LIBS="-lm -lgomp" && \
    strip miniAI

# Stage 2: Runtime
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="miniAI"
LABEL org.opencontainers.image.description="Feed-forward neural network in pure C without ML libraries."
LABEL org.opencontainers.image.authors="Nelson Ramos"
LABEL org.opencontainers.image.source="https://github.com/nelsonramosua/miniAI"

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy binary and required files from builder
COPY --from=builder /build/miniAI .
COPY --from=builder /build/IO ./IO
COPY --from=builder /build/tools ./tools
COPY --from=builder /build/README.md .
COPY --from=builder /build/LICENSE .

# Create directories for models and configs
RUN mkdir -p IO/models IO/configs IO/images

# Set permissions
RUN chmod +x miniAI

# Create non-root user
RUN useradd -m miniaiuser && \
    chown -R miniaiuser:miniaiuser /app
USER miniaiuser

# Default command
ENTRYPOINT ["./miniAI"]
CMD ["help"]

# Example usage:
# docker build -t miniai .
# docker run --rm miniai help
# docker run --rm miniai train --dataset digits --static
# docker run --rm -v $(pwd)/models:/app/IO/models miniai train --dataset digits --static