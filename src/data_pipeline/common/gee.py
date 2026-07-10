from __future__ import annotations


def initialize_gee(project_id: str | None = None):
    import ee

    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    return ee
