FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=on \
    DATA_DIR=/data

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml README.md ./
RUN uv pip install --system .

COPY src ./src

# Ensure data dir exists
RUN mkdir -p ${DATA_DIR}

# Copy start script
COPY start.sh ./start.sh

EXPOSE 8000 8501

CMD ["/bin/sh", "-c", "chmod +x ./start.sh && ./start.sh"]


