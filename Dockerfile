# DeepSeek Discord Bot — container image
# Build:  docker build -t deepseek-discord-bot .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# Run as a non-root user for security.
RUN useradd --create-home --shell /usr/sbin/nologin botuser \
    && chown -R botuser:botuser /app
USER botuser

# docker-compose.yml passes the .env file; a standalone run can also use -e flags.
CMD ["python", "bot.py"]
