from app import app


def test_home_page_presents_research_scope_and_process_navigation() -> None:
    response = app.test_client().get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "EWMA Covariate Shift Research Hub" in html
    assert "Local RAG support layer" in html
    assert 'href="/support"' in html
    assert "Current project progress" in html
    assert "Detection through confidence-gated transductive adaptation is" in html
    assert "Dynamic CSE-UAEL ensemble growth" in html
    assert "Update policy" in html
    assert "Label acquisition" in html
    assert "Transductive adaptation" in html
    assert "Dynamic ensemble" in html
    assert "planned" in html
    assert "Research process" in html
    assert "Covariate shift" in html
    assert "Validation" in html
    assert "Adaptive learning" in html
    assert 'href="/process/covariate-shift"' in html
    assert 'href="/process/validation"' in html
    assert 'href="/process/adaptive-learning"' in html
    assert "Table 1 dashboard" in html
    assert 'href="/table-1"' in html
    assert 'href="/results"' in html
    assert 'href="/figure-1"' in html
    assert 'href="/chowdhury-demographics"' in html
    assert 'href="/tests"' in html
    assert 'href="/papers/2015-raza-ewma-covariate-shift.pdf"' in html
    assert 'href="/papers/2018-chowdhury-online-adaptive-bci.pdf"' in html
    assert 'href="/papers/2019-raza-cse-uael.pdf"' in html
    assert html.count("Preview paper PDF") == 3


def test_process_pages_and_results_have_separate_destinations() -> None:
    client = app.test_client()
    for stage in ("covariate-shift", "validation", "adaptive-learning"):
        response = client.get(f"/process/{stage}")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert 'aria-label="Inputs and outputs"' in html
        assert 'aria-label="Method steps"' in html
        assert 'href="/results"' in html
        assert "Published CSV mean" not in html
        assert "Subject-level accuracy" not in html
        if stage == "adaptive-learning":
            assert 'aria-label="Continual learning dimensions"' in html
            assert "Active adaptation ≠ active learning" in html
            assert "PeriodicRetrain" in html
            assert "ContinuousRetrain" in html
            assert "RetrainOnValidatedShift" in html
            assert 'href="/results/transductive-adaptation"' in html
    assert client.get("/process/unknown").status_code == 404
    results = client.get("/results").get_data(as_text=True)
    assert 'aria-label="Result pages by process"' in results
    assert 'href="/results/paper2015-d2/ks-validation"' in results
    assert 'href="/results/diethe"' in results
    assert 'href="/results/transductive-adaptation"' in results
    assert "Continual learning and classifier evidence" in results
    assert 'href="/results/data-processing"' in results

    transductive = client.get("/results/transductive-adaptation")
    assert transductive.status_code == 200
    transductive_html = transductive.get_data(as_text=True)
    assert "Transductive adaptation event sequence" in transductive_html
    assert "Implementation evidence is not model-performance evidence" in transductive_html
    assert "pseudo_label_candidates" in transductive_html



def test_paper_preview_serves_repository_pdf_inline() -> None:
    response = app.test_client().get(
        "/papers/2015-raza-ewma-covariate-shift.pdf"
    )

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF-")
    assert response.headers["Content-Disposition"].startswith("inline;")



def test_paper_preview_rejects_non_pdf_files() -> None:
    response = app.test_client().get("/papers/README.md")

    assert response.status_code == 404
