import json
import sys
import numpy as np
from scripts.run_gpr_kernel_validation import main


def test_prepared_cli_exports_finite_auditable_evidence(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    training = rng.normal(size=(60,2))
    inputs = tmp_path / 'prepared.npz'
    np.savez(inputs, training_features=training, training_targets=training[:,0],
             calibration_features=rng.normal(size=(100,2)),
             evaluation_features=rng.normal(size=(80,2)), evaluation_times=np.arange(80))
    provenance = tmp_path / 'source.json'
    provenance.write_text(json.dumps(dict(dataset='synthetic fixture',subject='fixture',
        split='independent arrays',preprocessing='none',target_definition='first feature')))
    output = tmp_path / 'output'
    monkeypatch.setattr(sys,'argv',['runner','--features',str(inputs),'--provenance',str(provenance),
                                   '--permutations','19','--output',str(output)])
    main()
    result = json.loads((output / 'validation.json').read_text())
    assert result['results'][0]['source']['dataset'] == 'synthetic fixture'
    assert len(result['source_sha256']) == 6
    assert result['revision']
