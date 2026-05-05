FROM apache/airflow:3.2.0-python3.14

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

ENV AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    PYTHONPATH=/opt/airflow/dags:/opt/airflow \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy \
    PATH=/opt/airflow/.venv/bin:$PATH

WORKDIR /opt/airflow

USER root
COPY --chown=airflow:0 docker/entrypoint.sh /entrypoint
RUN chmod +x /entrypoint \
    && mkdir -p /opt/airflow/.dlt \
    && chown -R airflow:0 /opt/airflow/.dlt

USER airflow
COPY --chown=airflow:0 pyproject.toml uv.lock /opt/airflow/
RUN uv sync --locked --no-dev --no-install-project

COPY --chown=airflow:0 data_loader/ /opt/airflow/dags/
COPY --chown=airflow:0 schemas/ /opt/airflow/schemas/
COPY --chown=airflow:0 .dlt/config.toml /opt/airflow/.dlt/config.toml

EXPOSE 8080
ENTRYPOINT ["/entrypoint"]
CMD ["airflow", "standalone"]
