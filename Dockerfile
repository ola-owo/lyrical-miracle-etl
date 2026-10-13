FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG DATA_DIR=/data
ENV PYTHONUNBUFFERED=1 \
    DAGSTER_HOME=/opt/dagster/home \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_LOCKED=1 \
    UV_NO_DEV=1

WORKDIR /opt/dagster/app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project
COPY data_loader/ ./data_loader/
RUN uv sync

COPY schemas/ ./schemas/
COPY .dlt/ ./.dlt/
COPY dagster.yaml entrypoint.sh /opt/dagster/home/

USER root
RUN useradd --create-home --uid 1000 --gid 0 dagster && \
    mkdir -p ${DATA_DIR} /opt/dagster/home/storage && \
    chmod +x /opt/dagster/home/entrypoint.sh && \
    chown -R dagster:0 /opt/dagster ${DATA_DIR}
USER dagster

# persist compute logs and data files across container restarts
VOLUME ["/opt/dagster/home/storage", "${DATA_DIR}"]

EXPOSE 3000
ENTRYPOINT ["/opt/dagster/home/entrypoint.sh"]
