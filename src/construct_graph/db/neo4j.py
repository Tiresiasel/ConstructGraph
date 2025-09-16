from typing import Any
from py2neo import Graph
from ..config import CONFIG


def get_graph(override: dict | None = None) -> Graph:
    """Create a Neo4j Graph client using CONFIG or overrides.

    Parameters
    ----------
    override: dict | None
        Optional keys: uri, user, password.
    """
    cfg = CONFIG.neo4j
    uri = (override or {}).get('uri', cfg.uri)
    user = (override or {}).get('user', cfg.user)
    password = (override or {}).get('password', cfg.password)
    return Graph(uri, auth=(user, password))


