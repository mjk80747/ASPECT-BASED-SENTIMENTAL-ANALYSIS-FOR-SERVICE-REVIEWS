# Testing

This project includes a pytest setup for the Flask app and a test database configuration that uses SQLite in memory instead of the default file database.

## Install test dependencies

```bash
pip install -r requirements.txt
```

## Run the test suite

```bash
pytest
pytest -v
pytest --cov=app --cov-report=term-missing
```

## Coverage interpretation

- Coverage shows the percentage of lines executed by the tests.
- A higher percentage means the app routes and helpers are exercised more completely.
- `--cov-report=term-missing` highlights lines that are currently untested so you can add focused regression tests.
- Keep the app factory and SQLite setup minimal to avoid affecting the production web app behavior.
