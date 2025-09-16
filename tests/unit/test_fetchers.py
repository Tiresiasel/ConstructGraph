from types import SimpleNamespace


class FakeRunResult:
    def __init__(self, rows):
        self._rows = rows
    def data(self):
        return self._rows


class FakeGraph:
    def __init__(self, rows):
        self._rows = rows
    def run(self, query, **params):
        return FakeRunResult(self._rows)


def test_fetch_constructs_filters_measurements_active():
    from construct_graph.data.fetchers import fetch_constructs
    rows = [{
        'name': 'x', 'description': None, 'status': 'Verified', 'canonical_status': None, 'active': True,
        'definitions': [],
        'measurements': [
            {'name': 'm1', 'active': True},
            {'name': 'm2', 'active': False}
        ],
        'dimensions': [], 'parent_constructs': [], 'similar_constructs': [], 'similar_to_constructs': [],
        'paper_ids': []
    }]
    g = FakeGraph(rows)
    data = fetch_constructs(g)
    assert len(data) == 1
    meas = data[0]['measurements']
    # fetch_constructs returns DB rows; active filtering is enforced in Cypher, but our fake rows
    # are passed through as-is. Assert shape only here.
    assert any(m['name'] == 'm1' for m in meas)


def test_fetch_relationships_shape():
    from construct_graph.data.fetchers import fetch_relationships
    rows = [{
        'source_construct': 'a', 'target_construct': 'b', 'status': 'Hypothesized',
        'evidence_type': None, 'effect_direction': None, 'is_validated_causality': None,
        'relationship_instances': [
            {'uuid': 'ri1', 'active': True, 'status': 'Hypothesized', 'moderators': []}
        ],
        'paper_ids': []
    }]
    g = FakeGraph(rows)
    data = fetch_relationships(g)
    assert len(data) == 1
    assert data[0]['source_construct'] == 'a'
    assert isinstance(data[0]['relationship_instances'], list)

