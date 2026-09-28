from pathlib import Path

from src.support.rag import LocalSupportRAG


def test_local_support_rag_retrieves_repository_and_dataset_inventory(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "loader.py").write_text(
        "def load_dataset_2a():\n    return 'BCI Competition IV Dataset 2A loader'\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data" / "raw" / "bci_competition_iv_2a"
    data_dir.mkdir(parents=True)
    (data_dir / "A01T.gdf").write_bytes(b"placeholder")
    (tmp_path / "README.md").write_text(
        "Dataset 2A uses official evaluation labels and FBCSP features.",
        encoding="utf-8",
    )

    rag = LocalSupportRAG(tmp_path)
    rag.build_index()

    hits = rag.retrieve("Where is Dataset 2A loaded and which GDF file is present?", top_k=5)

    assert hits
    assert any(hit.chunk.source == "src/loader.py" for hit in hits)
    assert any(hit.chunk.kind == "dataset-inventory" for hit in hits)


def test_local_support_rag_uses_retrieved_context_for_generation(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "Bagging uses thirty estimators and a sample fraction of 0.8.",
        encoding="utf-8",
    )
    rag = LocalSupportRAG(tmp_path, model="llama3.2:1b")
    rag.build_index()

    captured = {}

    def fake_generator(prompt: str) -> str:
        captured["prompt"] = prompt
        return "The repository states 30 estimators [S1]."

    result = rag.answer("How many bagging estimators are used?", generator=fake_generator)

    assert result.answer == "The repository states 30 estimators [S1]."
    assert "[S1]" in captured["prompt"]
    assert "thirty estimators" in captured["prompt"]
    assert result.model_error is None
