from src.web import app


def test_page_empty_state_and_real_run():
    client = app.test_client()
    page = client.get('/results/diethe')
    assert page.status_code == 200
    assert b'Ready to compare' in page.data
    page = client.post('/results/diethe', data={'trials':80,'scenario':'relationship_shift'})
    assert page.status_code == 200
    assert b'experiment-data' in page.data
    assert b'performance_drop' in page.data
    assert b'not EEG' in page.data
    assert b'Download complete evidence JSON' in page.data
    assert b'Policy family' in page.data
    assert b'non-adaptive' in page.data
    assert b'passive' in page.data
    assert b'active' in page.data
    assert b'Trials since previous update' in page.data


def test_bad_inputs_do_not_run_unbounded_experiment():
    client = app.test_client()
    for data in ({'trials':100000}, {'seed':'bad'}, {'magnitude':'nan'}, {'update_scope':'unknown'}):
        response = client.post('/results/diethe', data=data)
        assert response.status_code == 400
        assert b'role="alert"' in response.data
        assert b'id="experiment-data"' not in response.data
