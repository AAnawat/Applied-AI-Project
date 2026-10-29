FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# models/sentiment.joblib is in Git and is copied in with the rest of the code.
# .streamlit/secrets.toml is mounted at run time by docker-compose.yml, never baked in.
COPY --chown=app . .
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"

# "python -m" puts the repository root on sys.path so dashboard/ can import categorization/ and sentiment/
CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
