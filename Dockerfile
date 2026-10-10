# Streamlit dashboard image. Also used as the Hugging Face Spaces (Docker SDK) image.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

# Spaces run containers as uid 1000; keep everything it writes under /app.
RUN useradd -m -u 1000 app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.hf-cache \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PORT=7860 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

COPY --chown=app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project && chown -R app /app

COPY --chown=app . .
USER app

EXPOSE 7860
CMD ["sh", "deploy/start.sh"]
