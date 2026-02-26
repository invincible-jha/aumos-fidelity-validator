.PHONY: install test test-quick lint format typecheck clean all docker-build docker-run migrate

all: lint typecheck test

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=aumos_fidelity_validator --cov-report=term-missing

test-quick:
	pytest tests/ -x -q --no-header

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/aumos_fidelity_validator/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info

docker-build:
	docker build -t aumos/fidelity-validator:dev .

docker-run:
	docker compose -f docker-compose.dev.yml up -d

migrate:
	alembic -c src/aumos_fidelity_validator/migrations/alembic.ini upgrade head
