FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY decisionforge ./decisionforge
COPY app ./app
RUN pip install --no-cache-dir -e ".[llm,app]"
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
