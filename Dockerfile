FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock* .

RUN pip install --no-cache-dir uv
RUN uv sync --no-dev

COPY . .

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]