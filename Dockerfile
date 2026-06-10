FROM apache/airflow:slim-3.2.1-python3.14

ARG DATA_DIR=/data \
    AIRFLOW_PORT=8080
ENV PYTHONUNBUFFERED=1 \
    AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__DAGS_FOLDER=${AIRFLOW_HOME}/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    AIRFLOW__LOGGING__BASE_LOG_FOLDER=${AIRFLOW_HOME}/logs \
    AIRFLOW__API__PORT=$AIRFLOW_PORT \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_LOCKED=1 \
    UV_NO_DEV=1

WORKDIR ${AIRFLOW_HOME}
COPY pyproject.toml uv.lock ./
RUN uv venv --system-site-packages && \
    uv sync --active --no-install-project
COPY data_loader/ ./data_loader/
RUN uv sync --active

ARG DAGS_FOLDER="airflow/dags"
COPY ${DAGS_FOLDER} ./dags/
COPY schemas/ ./schemas/
COPY .dlt/ ./.dlt/

USER root
RUN mkdir -p ${AIRFLOW__LOGGING__BASE_LOG_FOLDER} ${DATA_DIR} && \
    chown -R airflow:0 ${AIRFLOW__LOGGING__BASE_LOG_FOLDER} ${DATA_DIR} && \
    chown -R airflow:0 ${AIRFLOW_HOME}
USER airflow

VOLUME ["${AIRFLOW__LOGGING__BASE_LOG_FOLDER}", "${DATA_DIR}"]

EXPOSE $AIRFLOW_PORT
