# Start the docker container
exec docker run --rm -it -p 8080:8080 --env-file .env \
  --mount type=volume,src=airflow-logs,dst=/opt/airflow/logs \
  --mount type=bind,src=$(pwd)/data,dst=/data,ro \
  --mount type=bind,src=$(pwd)/keys,dst=/opt/airflow/keys,ro \
  --mount type=bind,src=$(pwd)/airflow/dags,dst=/opt/airflow/dags,ro \
  lyrical-miracle-etl standalone
