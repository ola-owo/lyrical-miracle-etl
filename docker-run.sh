# Start the docker container
exec docker run --rm -it -p 3000:3000 --env-file .env \
  --mount type=volume,src=dagster-logs,dst=/opt/dagster/home/storage \
  --mount type=bind,src=$(pwd)/data,dst=/data,ro \
  --mount type=bind,src=$(pwd)/keys,dst=/opt/dagster/app/keys,ro \
  lyrical-miracle-etl "$@"
