FROM apache/airflow:3.2.1-python3.14

ARG DATA_DIR=/data
ENV AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__DAGS_FOLDER=${AIRFLOW_HOME}/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    AIRFLOW__LOGGING__BASE_LOG_FOLDER=${AIRFLOW_HOME}/logs \
    UV_LINK_MODE=copy

WORKDIR ${AIRFLOW_HOME}
COPY pyproject.toml uv.lock ./
RUN uv venv --system-site-packages && \
    uv sync -n --active --locked --no-dev --no-install-project
COPY data_loader/ ./data_loader/
RUN uv sync -n --active --locked --no-dev

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

EXPOSE 8080
