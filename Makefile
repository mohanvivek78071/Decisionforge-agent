.PHONY: install test eval demo app clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

eval:
	python -m decisionforge eval

demo:
	python -m decisionforge ask "Why did revenue decline recently and where exactly is it coming from?"

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf runs .pytest_cache **/__pycache__
