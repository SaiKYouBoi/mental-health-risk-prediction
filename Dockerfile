FROM python:3.11-slim AS streamlit
WORKDIR /app
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt
COPY app ./app
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]

FROM apache/airflow:3.4.0-python3.11 AS airflow
WORKDIR /app
COPY --chown=airflow:root requirements.txt ./
RUN pip install --no-cache-dir "apache-airflow==3.4.0" -r requirements.txt \
    && mkdir -p /opt/airflow/state \
    && chmod 775 /opt/airflow/state
COPY --chown=airflow:root src ./src
COPY --chown=airflow:root dags /opt/airflow/dags
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONPATH=/app
CMD ["standalone"]
