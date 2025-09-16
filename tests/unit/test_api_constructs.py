from types import SimpleNamespace


class FakeRunResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class FakeTx:
    def __init__(self, run_map=None):
        self.run_map = run_map or {}

    def run(self, query, **params):
        if 'SET cc.active = false' in query:
            name = (params.get('n') or '')
            return FakeRunResult([{"name": name, "active": False}])
        return FakeRunResult(self.run_map.get('default', []))

    def commit(self):
        return None

    def rollback(self):
        return None


class FakeGraph:
    def __init__(self, run_map=None):
        self.run_map = run_map or {}

    def run(self, query, **params):
        return FakeRunResult(self.run_map.get('default', []))

    def begin(self):
        return FakeTx(self.run_map)


def make_app(monkeypatch, constructs_rows=None):
    from server import app as app_module

    # Patch DB client before app creation
    monkeypatch.setattr(app_module, 'get_graph_client', lambda: FakeGraph())

    # Patch data fetchers
    def fake_fetch_constructs(graph):
        return constructs_rows if constructs_rows is not None else []

    monkeypatch.setattr(app_module, 'fetch_constructs', fake_fetch_constructs)

    # Stub backend ops
    monkeypatch.setattr(app_module, 'update_construct_description', lambda *args, **kwargs: {
        'revisions': [{'revision_id': 'r1'}], 'embedding': {'dimension': 1024}, 'candidates': [], 'merges': []
    })
    monkeypatch.setattr(app_module, '_apply_soft_merge', lambda *args, **kwargs: 'op123')
    monkeypatch.setattr(app_module, 'rollback_merge', lambda *args, **kwargs: {'rolled_back': True})

    app = app_module.create_app()
    app.testing = True
    return app


def test_constructs_list_pagination_and_filters(monkeypatch):
    rows = [
        {'name': 'alpha network', 'active': True},
        {'name': 'beta link', 'active': False},
        {'name': 'alliance synergy', 'active': True},
    ]
    app = make_app(monkeypatch, constructs_rows=rows)
    client = app.test_client()

    # Default active=true
    resp = client.get('/api/constructs')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'total' in data and 'items' in data
    assert data['total'] == 2
    assert len(data['items']) == 2

    # q filter
    resp = client.get('/api/constructs?q=alli')
    data = resp.get_json()
    assert data['total'] == 1
    assert data['items'][0]['name'] == 'alliance synergy'

    # include inactive
    resp = client.get('/api/constructs?active=false')
    data = resp.get_json()
    assert data['total'] == 3


def test_update_construct_description_endpoint(monkeypatch):
    app = make_app(monkeypatch, constructs_rows=[])
    client = app.test_client()

    resp = client.patch('/api/constructs/alpha%20network/description', json={
        'description': 'Updated definition',
        'auto_merge': False,
        'editor': 'tester'
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert 'revisions' in body
    assert body['revisions'][0]['revision_id'] == 'r1'


def test_merge_and_rollback_endpoints(monkeypatch):
    app = make_app(monkeypatch, constructs_rows=[])
    client = app.test_client()

    # merge
    r = client.post('/api/constructs/merge', json={'keep': 'a', 'drop': 'b', 'similarity': 0.9, 'confidence': 0.96})
    assert r.status_code == 200
    assert r.get_json()['operation_id'] == 'op123'

    # rollback
    r = client.post('/api/constructs/rollback-merge', json={'operation_id': 'op123'})
    assert r.status_code == 200
    assert r.get_json()['rolled_back'] is True


def test_soft_delete_construct(monkeypatch):
    from server import app as app_module
    # Graph with begin().run returning a row for delete
    fake_graph = FakeGraph(run_map={'default': [{'ok': 1}]})
    monkeypatch.setattr(app_module, 'get_graph_client', lambda: fake_graph)
    # Keep other patches minimal
    monkeypatch.setattr(app_module, 'fetch_constructs', lambda graph: [])
    app = app_module.create_app()
    app.testing = True
    client = app.test_client()

    r = client.delete('/api/constructs/alpha%20network')
    assert r.status_code == 200
    body = r.get_json()
    assert body['active'] is False

