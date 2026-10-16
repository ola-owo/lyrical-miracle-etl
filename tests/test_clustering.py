"""Tests for data_loader.clustering helpers."""

import numpy as np
import polars as pl
import pytest

from data_loader.clustering import (
    build_kmeans_from_centroids,
    kmeans_to_dataframe,
    run_kmeans,
)

# two well-separated groups of 2D points
EMBEDDINGS = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.0],
        [10.0, 10.0],
        [10.1, 10.0],
    ]
)
CENTROIDS = np.array([[0.0, 0.0], [10.0, 10.0]])


def test_run_kmeans_separates_well_separated_groups():
    km = run_kmeans(EMBEDDINGS, n_clusters=2, random_seed=0)

    assert km.cluster_centers_.shape == (2, 2)
    assert len(set(km.labels_[:2])) == 1
    assert len(set(km.labels_[2:])) == 1
    assert km.labels_[0] != km.labels_[2]


def test_run_kmeans_is_deterministic():
    labels_1 = run_kmeans(EMBEDDINGS, n_clusters=2, random_seed=0).labels_
    labels_2 = run_kmeans(EMBEDDINGS, n_clusters=2, random_seed=0).labels_
    assert labels_1.tolist() == labels_2.tolist()


def test_build_kmeans_from_centroids_preserves_centroids():
    km = build_kmeans_from_centroids(CENTROIDS)
    assert km.cluster_centers_.tolist() == CENTROIDS.tolist()


def test_build_kmeans_from_centroids_assigns_nearest_cluster():
    km = build_kmeans_from_centroids(CENTROIDS)
    points = np.array([[0.2, 0.1], [9.9, 10.0]])
    assert km.predict(points).tolist() == [0, 1]


def test_kmeans_to_dataframe_returns_labels_and_centroids():
    km = build_kmeans_from_centroids(CENTROIDS)

    res = kmeans_to_dataframe(km, [7, 9], [0, 1])

    labels = res['labels']
    assert labels.schema == {'id': pl.Int64, 'cluster': pl.Int64}
    assert labels['id'].to_list() == [7, 9]
    assert labels['cluster'].to_list() == [0, 1]

    centroids = res['centroids']
    assert centroids.schema == {'cluster': pl.Int64, 'centroid': pl.List(pl.Float64)}
    assert centroids['cluster'].to_list() == [0, 1]
    assert centroids['centroid'].to_list() == [[0.0, 0.0], [10.0, 10.0]]


def test_kmeans_to_dataframe_rejects_mismatched_lengths():
    km = build_kmeans_from_centroids(CENTROIDS)
    with pytest.raises(AssertionError):
        kmeans_to_dataframe(km, [1, 2], [0])
