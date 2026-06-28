# Multi-stage Docker build for agent-memory
ARG VERSION=0.0.0
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY agent_memory/ ./agent_memory/
COPY mcp_server/ ./mcp_server/

# Set version for hatch-vcs (since .git is not available in build context)
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AGENT_MEMORY=${VERSION}

# Install runtime dependencies including chromadb
RUN pip install --no-cache-dir -e ".[chromadb]"

# Runtime stage - use slim for pip availability, but keep it minimal
FROM python:3.11-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory to user's home directory
WORKDIR /home/appuser

# Copy installed packages from builder (only runtime deps)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code and install in runtime (without dev dependencies)
COPY --from=builder /app/agent_memory ./agent_memory
COPY --from=builder /app/mcp_server ./mcp_server
COPY --from=builder /app/pyproject.toml ./pyproject.toml
COPY --from=builder /app/README.md ./README.md

# Install package in runtime (without dev dependencies)
RUN pip install --no-cache-dir --no-deps -e .

# Create data directory for persistence
RUN mkdir -p /home/appuser/.agent_memory && chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser
ENV HOME=/home/appuser
ENV AGENT_MEMORY_DIR=/home/appuser/.agent_memory

# Expose MCP server port (if needed)
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["agent-memory"]