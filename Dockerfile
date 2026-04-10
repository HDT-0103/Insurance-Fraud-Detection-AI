FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.prod.txt ./requirements.prod.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.prod.txt

COPY src ./src
COPY app.py ./app.py
COPY models ./models
COPY data ./data
COPY pyproject.toml ./pyproject.toml
COPY start.sh ./start.sh

RUN chmod +x /app/start.sh

EXPOSE 8501

CMD ["/app/start.sh"]

