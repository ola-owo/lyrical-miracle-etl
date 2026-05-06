FROM apache/airflow:3.2.0-python3.14

ENV \
    AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    UV_LINK_MODE=copy

WORKDIR /opt/airflow
COPY pyproject.toml uv.lock ./
RUN uv venv --system-site-packages && uv sync -n --active --locked --no-dev --no-install-project
COPY data_loader/ ./data_loader/
RUN uv sync -n --active --locked

COPY dags/ ./dags/
COPY schemas/ ./schemas/
COPY .dlt/ ./.dlt/

USER root
RUN chown -R airflow:0 .
USER airflow

EXPOSE 8080
