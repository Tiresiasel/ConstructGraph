class FakeRunResult:
    def __init__(self, rows):
        self._rows = rows
    def data(self):
        return self._rows


class FakeTx:
    def __init__(self):
        pass
    def run(self, query, **params):
        # Return something plausible for soft-delete/restore
        if 'SET m.active = false' in query:
            return FakeRunResult([{"uuid": params.get('uuid'), "active": False}])
        if 'SET d.active = false' in query:
            return FakeRunResult([{"uuid": params.get('uuid'), "active": False}])
        if 'MATCH (m:Measurement {uuid:' in query and 'SET' not in query:
            return FakeRunResult([{"uuid": params.get('uuid')}])
        return FakeRunResult([])
    def commit(self):
        return None
    def rollback(self):
        return None


class FakeGraph:
    def run(self, query, **params):
        # relationships listing returns empty array by default
        return FakeRunResult([])
    def begin(self):
        return FakeTx()


def make_app(monkeypatch):
    from server import app as app_module
    # Patch DB client before app creation
    monkeypatch.setattr(app_module, 'get_graph_client', lambda: FakeGraph())
    # Patch fetchers
    monkeypatch.setattr(app_module, 'fetch_relationships', lambda graph: [
        {
            'source_construct': 'a',
            'target_construct': 'b',
            'status': 'Empirical_Result',
            'evidence_type': 'Quantitative',
            'relationship_instances': [
                {'uuid': 'ri1', 'status': 'Empirical_Result', 'evidence_type': 'Quantitative', 'active': True}
            ]
        }
    ])
    # Patch backend ops
    monkeypatch.setattr(app_module, 'create_relationship_instance', lambda *args, **kwargs: {'ri_uuid': 'ri_new'})
    monkeypatch.setattr(app_module, 'update_relationship_instance', lambda *args, **kwargs: {
        'operation_id': 'rop1', 'rewires': [], 'updated_fields': list((kwargs.get('props') or {}).keys())
    })
    monkeypatch.setattr(app_module, 'soft_delete_relationship_instance', lambda *args, **kwargs: {'active': False})
    monkeypatch.setattr(app_module, 'restore_relationship_instance', lambda *args, **kwargs: {'active': True})
    monkeypatch.setattr(app_module, 'rollback_relationship_operation', lambda *args, **kwargs: {'rolled_back': True})

    app = app_module.create_app()
    app.testing = True
    return app


def test_relationships_list_filters(monkeypatch):
    app = make_app(monkeypatch)
    client = app.test_client()

    r = client.get('/api/relationships?status=Empirical_Result')
    assert r.status_code == 200
    body = r.get_json()
    assert 'items' in body and body['total'] == 1
    assert body['items'][0]['status'] == 'Empirical_Result'

    r = client.get('/api/relationships?evidence_type=Quantitative')
    body = r.get_json()
    assert body['total'] == 1


def test_relationship_create_update_soft_delete_restore(monkeypatch):
    app = make_app(monkeypatch)
    client = app.test_client()

    # create
    r = client.post('/api/relationships', json={'subject': 'a', 'object': 'b', 'status': 'Hypothesized'})
    assert r.status_code == 200
    assert r.get_json()['ri_uuid'] == 'ri_new'

    # update props
    r = client.patch('/api/relationships/ri1', json={'props': {'effect_direction': 'Positive'}})
    assert r.status_code == 200
    assert 'operation_id' in r.get_json()

    # soft delete
    r = client.post('/api/relationships/ri1/soft-delete', json={})
    assert r.status_code == 200
    assert r.get_json()['active'] is False

    # restore
    r = client.post('/api/relationships/ri1/restore', json={})
    assert r.status_code == 200
    assert r.get_json()['active'] is True

    # rollback operation
    r = client.post('/api/relationships/rollback-operation', json={'operation_id': 'rop1'})
    assert r.status_code == 200
    assert r.get_json()['rolled_back'] is True

