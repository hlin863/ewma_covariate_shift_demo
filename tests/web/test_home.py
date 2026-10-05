from app import app


def test_home_page_presents_research_scope_and_application_map() -> None:
    response = app.test_client().get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "EWMA Covariate Shift Research Hub" in html
    assert "Local RAG support layer" in html
    assert 'href="/support"' in html
    assert "Current project progress" in html
    assert "Detection through confidence-gated transductive adaptation is" in html
    assert "Dynamic CSE-UAEL ensemble growth" in html
    assert "Transductive adaptation" in html
    assert "Dynamic ensemble" in html
    assert "planned" in html
    assert "Application architecture" in html
    assert "Data and protocol foundation" in html
    assert "Detection and representation analysis" in html
    assert "Adaptation and continual learning" in html
    assert "Evidence and implementation assurance" in html
    assert "Diethe policy laboratory" in html
    assert 'href="/results/diethe"' in html
    assert "reads across all layers" in html
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
