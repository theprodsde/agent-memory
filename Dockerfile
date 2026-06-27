# Multi-stage Docker build for agent-memory
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

# Install package in development mode
RUN pip install --no-cache-dir -e ".[dev]"

# Runtime stage
FROM python:3.11-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY --from=builder /app/agent_memory ./agent_memory
COPY --from=builder /app/mcp_server ./mcp_server

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

# Default command
ENTRYPOINT ["agent-memory"]
CMD ["--help"]