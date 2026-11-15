# syntax=docker/dockerfile:1.7
# Multi-stage: build the app's wheel once, ship a slim image without PyTorch that runs as a non-root user.
# The collection is not in the image: mount it at /collection (Lab/assets/lens, with its index built).
FROM python:3.12-slim AS build
WORKDIR /src
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml ./
COPY main.py ./
COPY app ./app
COPY config ./config
RUN pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8030 COLLECTION_DIR=/collection
RUN useradd --create-home --uid 10001 app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
USER app
WORKDIR /home/app
EXPOSE 8030
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=3)"
# One worker keeps the in-memory collection aligned with uploads on disk
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --no-access-log"]
