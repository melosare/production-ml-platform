ML Platform
A production-oriented machine learning platform: develop, test, containerize, and deploy

Status: Early development — platform foundation established; core ML functionality is under active development.

Objectives
The platform will provide an end-to-end foundation for machine learning workflows, including:

- Data ingestion and validation
- Feature engineering 
- Model training and evaluation 
- Batch and online inference 
- Model and experiment management 
- Monitoring and observability 
- Reproducible development and deployment 
- Automated testing and continuous integration

Architecture
The intended high-level architecture is:

                    Data Sources
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
              ┌──────────┴──────────┐
              ▼                     ▼
         Model Training       Batch Inference
              │                     │
              ▼                     │
        Model Evaluation            │
              │                     │
              ▼                     │
         Model Registry             │
              │                     │
              └──────────┬──────────┘
                         ▼
                  Model Serving
                         │
                         ▼
                    Monitoring

The architecture will evolve as individual components are implemented.

Development Environment
The project currently targets:

Python 3.11
Conda
PyCharm
Ruff
mypy
pytest
pre-commit
Docker
Docker Compose
GitHub Actions

The Python project configuration is maintained in pyproject.toml

Testing
Run the test suite with:

pytest

Run the test suite inside Docker:

docker compose run --rm test

The Docker-based test environment provides an additional check that the project can execute independently of the local Conda environment.

Code Quality
Format the project:

ruff format .

Check formatting without modifying files:

ruff format --check .

Run linting:

ruff check .

Run static type checking:

mypy src

Run all pre-commit hooks:

pre-commit run --all-files

Continuous Integration
GitHub Actions is configured to run the project's quality checks on pushes to main and pull requests targeting main.

The CI pipeline currently performs:

Python environment setup

Project and development dependency installation

Ruff formatting validation

Ruff linting

mypy type checking

pytest execution

The goal is to ensure that code entering the main branch satisfies the project's baseline quality requirements.
