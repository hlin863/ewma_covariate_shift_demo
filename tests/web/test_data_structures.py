from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from app import app


def test_overview_links_to_each_dataset():
    html = app.test_client().get('/data-structures').get_data(as_text=True)
    assert html.count('class="dataset-card"') == 6
    for link in ['/data-distributions?dataset=2a', '/data-distributions?dataset=2b',
                 '/data-structures/turbofan', '/data-structures/algae',
                 '/chowdhury-demographics', '/data-structures/synthetic']:
        assert f'href="{link}"' in html


@pytest.mark.parametrize('path', ['/', '/table-1', '/results', '/data-processing',
                                  '/results/diethe', '/results/decision-models',
                                  '/data-distributions?dataset=2b', '/chowdhury-demographics'])
def test_dataset_navigation_is_available_across_pages(path, tmp_path):
    app.config.update(DATASET_2A_PATH=str(tmp_path), DATASET_2B_PATH=str(tmp_path))
    response = app.test_client().get(path)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'aria-label="Dataset subpages"' in html
    assert 'href="/data-structures"' in html
    assert 'href="/data-distributions?dataset=2b"' in html


@pytest.mark.parametrize('subset, engines', [('FD001', '100'), ('FD002', '260'),
                                          ('FD003', '100'), ('FD004', '249')])
def test_turbofan_inspector_uses_loaded_data(subset, engines):
    app.config['TURBOFAN_DATA_PATH'] = str(Path(__file__).resolve().parents[2] / 'data/raw/turbofan_engine_degradation')
    response = app.test_client().get('/data-structures/turbofan', query_string={'subset': subset})
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert f'<strong>{engines}</strong>' in html
    assert 'sensor_21' in html
    assert 'Training observations with derived RUL' in html
    assert 'Settings and sensor distributions' in html
    assert 'Test observations with offline RUL' in html


def test_algae_inspection_preserves_independent_sampling(tmp_path):
    path = tmp_path / 'algae.mat'
    savemat(path, {'algae': np.array([
        {'temperature': {'data': [20, 21, 22], 'time_num': [0, .5, 1]},
         'pH': {'data': [7.2, np.nan], 'time_num': [0, 1]}},
        {'temperature': {'data': [23], 'time_num': [2]},
         'pH': {'data': [8], 'time_num': [2]}},
    ], dtype=object)})
    app.config['ALGAE_DATA_PATH'] = str(path)
    client = app.test_client()
    html = client.get('/data-structures/algae').get_data(as_text=True)
    assert 'Raceway 1 measurement fields' in html
    assert '<th>temperature</th><td>3</td>' in html
    assert '<th>pH</th><td>2</td>' in html
    assert 'Missing / non-finite' in html
    html = client.get('/data-structures/algae?raceway=2').get_data(as_text=True)
    assert 'Raceway 2 measurement fields' in html
    assert '<th>temperature</th><td>1</td>' in html
    assert client.get('/data-structures/algae?raceway=999').status_code == 400


def test_synthetic_page_reads_configured_csv(tmp_path):
    path = tmp_path / 'synthetic.csv'
    path.write_text('time,x,true_regime,true_shift\n0,1.5,before,0\n1,4.5,after,1\n')
    app.config['SYNTHETIC_DATA_PATH'] = str(path)
    response = app.test_client().get('/data-structures/synthetic')
    assert response.status_code == 200
    assert b'true_regime' in response.data
    assert b'1.5' in response.data


@pytest.mark.parametrize('dataset, config', [('turbofan', 'TURBOFAN_DATA_PATH'),
                                         ('algae', 'ALGAE_DATA_PATH'),
                                         ('synthetic', 'SYNTHETIC_DATA_PATH')])
def test_missing_data_keeps_dataset_navigation(dataset, config, tmp_path):
    app.config[config] = str(tmp_path / 'missing')
    response = app.test_client().get(f'/data-structures/{dataset}')
    assert response.status_code == 200
    assert b'Source data could not be loaded' in response.data
    assert b'Dataset subpages' in response.data


def test_invalid_dataset_and_subset_do_not_load_other_sources():
    client = app.test_client()
    assert client.get('/data-structures/unknown').status_code == 404
    assert client.get('/data-structures/turbofan?subset=FD005').status_code == 400
