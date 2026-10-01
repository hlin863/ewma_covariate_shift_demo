from app import app
from src.web.support import clear_support_cache


def test_support_page_is_available() -> None:
    app.config.update(
        TESTING=True,
        SUPPORT_RAG_PROPOSAL_PATH="",
        SUPPORT_RAG_MODEL="llama3.2:1b",
    )
    clear_support_cache()

    response = app.test_client().get("/support")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Ask the repository about its code, datasets and research design." in html
    assert "llama3.2:1b" in html
    assert "Raw GDF/MAT signal contents are not sent to the LLM." in html
