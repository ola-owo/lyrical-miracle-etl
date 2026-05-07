import dlt


def get_normalize_row_counts(pipeline: dlt.Pipeline) -> dict[str, int]:
    """Return normalized row counts from a dlt pipeline trace, if available."""
    trace = pipeline.last_trace
    if trace is None:
        return {}

    normalize_info = trace.last_normalize_info
    if normalize_info is None:
        return {}

    return normalize_info.row_counts or {}
