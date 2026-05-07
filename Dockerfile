FROM apache/airflow:3.2.1-python3.14

ENV AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__DAGS_FOLDER=${AIRFLOW_HOME}/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    UV_LINK_MODE=copy

WORKDIR ${AIRFLOW_HOME}
COPY pyproject.toml uv.lock ./
RUN uv venv --system-site-packages && \
    uv sync -n --active --locked --no-dev --no-install-project
COPY data_loader/ ./data_loader/
RUN uv sync -n --active --locked

ARG DAGS_FOLDER="airflow/dags"
COPY ${DAGS_FOLDER} ./dags/
COPY schemas/ ./schemas/
COPY .dlt/ ./.dlt/

USER root
RUN chown -R airflow:0 .
USER airflow

EXPOSE 8080
