from app import app


def test_laboratory_is_navigable_and_has_no_prefilled_claims():
    client = app.test_client()
    response = client.get('/results/gpr-kernel')
    assert response.status_code == 200
    assert b'Run diagnostics' in response.data
    assert b'Observed results' not in response.data
    assert b'/results/gpr-kernel' in client.get('/results/diethe').data


def test_run_and_invalid_request():
    client = app.test_client()
    result = client.post('/results/gpr-kernel',data={'scenario':'mean_shift','seed':'42'})
    assert result.status_code == 200
    assert b'Observed results' in result.data
    assert b'Every candidate warning' in result.data
    invalid = client.post('/results/gpr-kernel',data={'scenario':'none','seed':'-1'})
    assert invalid.status_code == 400
