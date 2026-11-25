.PHONY: install lint format typecheck test check run docker

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy

test:
	pytest

check: lint typecheck test

run:
	uvicorn gateway.main:create_app --factory --reload

docker:
	docker compose up --build
