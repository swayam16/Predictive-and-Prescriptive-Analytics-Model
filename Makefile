.PHONY: install run test lint app

install:        ## Install the package with dev and app extras
	pip install -e ".[dev,app]"

run:            ## Simulate data, train, evaluate, write reports/ and artifacts/
	churnpolicy run

test:           ## Unit, leakage and end-to-end tests
	pytest

lint:
	ruff check . && ruff format --check .

app:            ## Interactive demo on http://localhost:8501
	streamlit run app/streamlit_app.py
