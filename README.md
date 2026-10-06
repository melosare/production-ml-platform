# MLOPs: Production ML Platform

A production-oriented machine learning platform demonstrating an end-to-end ML lifecycle in Python: data ingestion, validation, feature engineering, reproducible training, evaluation, inference, API serving, Docker, observability, and CI/CD.

## Status

**Production-platform foundation complete.**

The project uses deterministic synthetic data and a baseline Logistic Regression model. The synthetic dataset is intentionally designed with strong class separation, so evaluation results should be treated as an engineering demonstration rather than evidence of real-world model performance.

## Architecture

```text
Configuration
     │
     ▼
Data Ingestion
     │
     ▼
Data Validation
     │
     ▼
Feature Engineering
     │
     ▼
Training Dataset
     │
     ▼
Train / Validation / Test Split
     │
     ▼
Baseline Model Training
     │
     ├──────────────► Validation Evaluation
     │
     ▼
Final Test Evaluation
     │
     ▼
Model + Metadata Persistence
     │
     ▼
Inference
     │
     ▼
Prediction Service
     │
     ▼
FastAPI
     │
     ├──────────────► Health / Readiness
     │
     ▼
Docker Runtime
     │
     ▼
CI/CD
     │
     ▼
Deployment Ready
```

## Project Structure

```text
production-ml-platform/
├── .github/workflows/       # GitHub Actions CI
├── artifacts/               # Generated model artifacts
├── configs/                 # Environment configuration
├── docs/                    # Data contract documentation
├── scripts/                 # Training and synthetic-data scripts
├── src/ml_platform/
│   ├── api/                 # FastAPI application
│   ├── config/              # Configuration loading
│   ├── data/                # Loading and validation
│   ├── evaluation/          # Evaluation and metrics
│   ├── features/            # Feature engineering
│   ├── inference/           # Model inference
│   ├── model/               # Model implementation/persistence
│   ├── service/             # Prediction service
│   └── training/            # Dataset, split, and training lifecycle
├── tests/
├── Dockerfile               # Production runtime image
├── Dockerfile.test          # Test image
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Development Environment

The project targets:

- Python 3.11
- Conda or another isolated Python environment
- PyCharm or another Python IDE
- Ruff
- mypy
- pytest
- pre-commit
- Docker
- Docker Compose
- GitHub Actions

Install the project and development dependencies with:

```bash
pip install -e ".[dev]"
```

## Configuration

Configuration is stored in YAML files under `configs/`.

### Development

```yaml
environment: development
log_level: DEBUG
model_path: artifacts/baseline_model.joblib
```

### Production

```yaml
environment: production
log_level: INFO
model_path: artifacts/baseline_model.joblib
```

The API can select a configuration file through:

```bash
export ML_PLATFORM_CONFIG=configs/development.yaml
```

If the environment variable is not set, the application defaults to:

```text
configs/development.yaml
```

## Data and Features

The platform uses deterministic synthetic event data for development and testing.

Events are validated against the project's data contract before feature engineering.

User-level features include:

- `event_count`
- `session_count`
- `page_view_count`
- `feature_usage_count`
- `purchase_count`
- `total_purchase_value`

The model uses only:

- `event_count`
- `session_count`
- `page_view_count`
- `feature_usage_count`

Purchase-related features are deliberately excluded from model inputs because they directly reveal the target and would introduce target leakage.

## ML Training Lifecycle

The training pipeline follows:

```text
Load events
    ↓
Build features
    ↓
Create training examples
    ↓
Train / validation / test split
    ↓
Train baseline model
    ↓
Evaluate on validation data
    ↓
Evaluate final model on test data
    ↓
Persist model and metadata
```

The split is:

- deterministic
- reproducible
- stratified
- configurable
- validated
- mutually exclusive
- exhaustive

The baseline model is scikit-learn Logistic Regression.

Generate the model artifact with:

```bash
python scripts/train_baseline_model.py
```

Generated model artifacts are intentionally not tracked in Git. CI regenerates the deterministic model before building the production Docker image.

## Evaluation

The platform currently reports:

- accuracy
- precision
- recall

The synthetic dataset currently produces very strong baseline results because the generated classes are intentionally well separated.

These results should not be interpreted as production model performance.

## Testing

Run the complete test suite:

```bash
pytest
```

Run a specific test:

```bash
pytest tests/integration/test_api.py
```

Run tests inside the Docker test environment:

```bash
docker compose run --rm test
```

## Code Quality

Format the project:

```bash
ruff format .
```

Check formatting:

```bash
ruff format --check .
```

Run linting:

```bash
ruff check .
```

Run static type checking:

```bash
mypy src
```

Run all pre-commit hooks:

```bash
pre-commit run --all-files
```

The combined Makefile quality check is:

```bash
make quality
```

## API

The application exposes a FastAPI service on port `8000`.

### Health

```http
GET /health
```

Example:

```json
{
  "status": "ok"
}
```

### Readiness

```http
GET /ready
```

Example:

```json
{
  "status": "ready"
}
```

### Prediction

```http
POST /predict
```

Example request:

```json
{
  "event_count": 20,
  "session_count": 3,
  "page_view_count": 12,
  "feature_usage_count": 2
}
```

Example response:

```json
{
  "predicted_class": 1,
  "probability": 0.97
}
```

Prediction inputs are validated as non-negative integers. Invalid request data results in standard FastAPI validation errors, while prediction failures return an HTTP 500 response without exposing internal exception details.

## Running Locally

Start the API with:

```bash
make run
```

Or:

```bash
uvicorn ml_platform.api.main:create_application --factory
```

The default development configuration points to:

```text
artifacts/baseline_model.joblib
```

## Docker

The production image contains only runtime dependencies and application files required to serve the model.

Build it with:

```bash
docker build -t production-ml-platform .
```

Run it with:

```bash
docker run --rm \
  -p 8000:8000 \
  -e ML_PLATFORM_CONFIG=configs/production.yaml \
  production-ml-platform
```

Verify the application:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

The production container runs as a non-root user.

Development/test services are available through Docker Compose:

```bash
docker compose build
docker compose run --rm test
docker compose up -d app
docker compose down
```

The runtime image does not install the project's development dependencies.

## Observability and Production Hardening

The API includes lightweight production-oriented safeguards:

- application logging
- environment-aware log levels
- request timing
- prediction logging
- health endpoint
- readiness endpoint
- graceful prediction failures
- validated API inputs
- non-root Docker execution
- separation of runtime and development dependencies
- deterministic model generation

The project intentionally does not introduce a full Prometheus/Grafana monitoring stack.

## CI/CD

GitHub Actions runs on pushes to `main` and pull requests targeting `main`.

The CI pipeline verifies:

```text
Ruff formatting
     ↓
Ruff linting
     ↓
mypy
     ↓
pytest
     ↓
Deterministic model training
     ↓
Production Docker build
     ↓
Docker test suite
     ↓
Production container startup
     ↓
/health smoke test
     ↓
/ready smoke test
```

The model artifact is generated during CI rather than committed to Git. This ensures a clean checkout can reproduce the deployable runtime image.

## Deployment Readiness

The repository is structured to support deployment as a containerized FastAPI service.

The production image includes:

- application source
- runtime dependencies
- configuration
- generated model artifact

The project intentionally stops short of providing cloud infrastructure, Kubernetes, Terraform, authentication, a model registry, or distributed training. Those are potential future extensions rather than requirements of the current platform.

## Engineering Principles

The project emphasizes:

- clean separation of responsibilities
- explicit data contracts
- type checking
- deterministic testing
- reproducible training
- prevention of target leakage
- small production-oriented components
- runtime/development dependency separation
- containerized execution
- automated quality gates
- honest interpretation of synthetic ML results

## License

This project is currently intended as a demonstration.
