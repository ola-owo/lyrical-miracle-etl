from airflow.sdk import dag

from data_loader.clustering import cluster_task


@dag(
    dag_id='cluster_all',
    description='Re-cluster all lyrics embeddings and generate new centroids',
    max_active_runs=1,
)
def cluster_all():
    cluster_task(replace=True)


cluster_all()
