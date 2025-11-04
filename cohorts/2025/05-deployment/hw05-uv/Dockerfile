FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

# Copy project metadata for dependency installation
COPY pyproject.toml uv.lock ./

# Install uv and sync dependencies (fastapi, uvicorn, etc.)
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir uv \
    && uv sync --frozen --no-cache

# Ensure the project venv is on PATH (when not using `uv run` explicitly)
ENV VIRTUAL_ENV=/code/.venv
ENV PATH="/code/.venv/bin:${PATH}"

# Copy the FastAPI service; model pipeline_v2.bin is already in the base image
COPY service.py ./

# Let the service know where the model is (explicit)
ENV MODEL_PATH=/code/pipeline_v2.bin

EXPOSE 8000

# Use `uv run` so the virtualenv is activated reliably
CMD ["uv", "run", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
