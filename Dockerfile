# syntax=docker/dockerfile:1.7
#
# FRAUD-OPS dashboard: FastAPI + static frontend, serving saved model scores.
#
#   docker build -t fraud-ops .                    # bundle downloaded from the GitHub Release
#   docker build --target test .                   # run the pytest suite in the image's environment
#   docker build --build-arg BUNDLE_SOURCE=local --build-context localbundle=./dist -t fraud-ops .
#
# Stages: frontend (Node) -> deps (Python venv) -> test | bundle -> runtime.
# Base images are pinned by digest so a rebuild uses the same OS layers.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
ARG NODE_IMAGE=node:22-alpine@sha256:1314fb30ec37592d5845e5a160d522c0e680972aa534549940b8d73bde88481a

# Demo data + models (scripts/package_demo_assets.py), published as a GitHub Release asset.
ARG BUNDLE_SOURCE=remote
ARG BUNDLE_NAME=fraud-ops-demo-v1
ARG BUNDLE_URL=https://github.com/KarenEmad-1/ibm-credit-card-fraud-detection/releases/download/demo-data-v1/fraud-ops-demo-v1.tar.gz
ARG BUNDLE_SHA256=a09c2e1142666dc560cad59356f69a1fbaa43afe28cef68f673f23c0332448ee


# ---------------------------------------------------------------- frontend
# The dashboard runs unbuilt from the repo (CDN imports). Here it gets local copies of
# Preact/htm, D3 and the fonts, so the image works with no third-party CDN at runtime.
FROM ${NODE_IMAGE} AS frontend
WORKDIR /build
COPY dashboard/package.json dashboard/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY dashboard/ ./
RUN node build.mjs /out


# ---------------------------------------------------------------- deps
# Runtime packages in a virtualenv, installed from a hashed lockfile: pip refuses any
# file whose hash differs. Only this layer changes when dependencies change.
FROM ${PYTHON_IMAGE} AS deps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-serve.lock /tmp/
# CatBoost declares its plotting/widget libraries as hard requirements, but loading a
# model, predicting and computing ShapValues never imports them: ~220 MB removed.
# (narwhals stays: scikit-learn imports it.)
RUN pip install --require-hashes --no-deps -r /tmp/requirements-serve.lock \
 && pip uninstall -y -q plotly matplotlib fonttools pillow contourpy cycler kiwisolver pyparsing graphviz


# ---------------------------------------------------------------- test
# The same environment plus pytest. CI builds this target; a failing test fails the build.
FROM deps AS test
COPY requirements-dev.lock /tmp/
RUN pip install --require-hashes --no-deps -r /tmp/requirements-dev.lock
WORKDIR /app
COPY src/ src/
COPY api/ api/
COPY tests/ tests/
RUN python -m pytest tests/ -q -p no:cacheprovider


# ---------------------------------------------------------------- bundle
FROM ${PYTHON_IMAGE} AS bundle-remote
ARG BUNDLE_URL
ADD ${BUNDLE_URL} /bundle.tar.gz

FROM ${PYTHON_IMAGE} AS bundle-local
ARG BUNDLE_NAME
# Only used with --build-context localbundle=./dist (before the Release exists)
COPY --from=localbundle ${BUNDLE_NAME}.tar.gz /bundle.tar.gz

FROM bundle-${BUNDLE_SOURCE} AS bundle
ARG BUNDLE_SHA256 BUNDLE_NAME
# The build fails unless the archive is exactly the published one
RUN echo "${BUNDLE_SHA256}  /bundle.tar.gz" | sha256sum -c - \
 && mkdir /demo && tar -xzf /bundle.tar.gz -C /demo --strip-components=1 \
 && cd /demo && python - <<'EOF'
import hashlib, json, sys
m = json.load(open("MANIFEST.json"))
bad = [f for f, v in m["files"].items()
       if hashlib.sha256(open(f, "rb").read()).hexdigest() != v["sha256"]]
sys.exit(f"checksum mismatch: {bad}" if bad else 0)
EOF


# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
# The last three give freed memory back to the OS instead of keeping it in allocator pools
# (-24% peak memory under a 512 MB limit, same responses)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=info \
    FRAUD_SPLITS=test \
    FRAUD_DASHBOARD_DIR=/demo/dashboard \
    FRAUD_FEATURES_DIR=/demo/features \
    FRAUD_MODELS_DIR=/demo/models \
    FRAUD_REPORTS_DIR=/app/reports \
    ARROW_DEFAULT_MEMORY_POOL=system \
    MALLOC_TRIM_THRESHOLD_=131072 \
    MALLOC_MMAP_THRESHOLD_=131072

# Unprivileged user; files stay owned by root, so the app can read but not modify them
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app

# Ordered from least to most frequently changed, so code edits reuse the cached layers
COPY --from=deps /opt/venv /opt/venv
COPY --from=bundle /demo /demo
COPY reports/*.csv reports/
COPY --from=frontend /out dashboard/
COPY src/ src/
COPY api/ api/

USER app
EXPOSE 8000
# /health only answers once the data is loaded; start-period covers the load
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)" || exit 1
CMD ["python", "-m", "api"]
