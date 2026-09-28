"""Lightweight local RAG over the repository, dataset inventory and proposal.

The support layer is intentionally separate from the scientific model. It
retrieves local project evidence with TF-IDF and sends only the retrieved
context to a local Ollama-hosted Llama 3.2 model. Raw EEG/MAT files are never
embedded or sent to the language model; only dataset inventory metadata is
indexed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".tex",
    ".html",
    ".csv",
}
_SOURCE_ROOTS = (
    "src",
    "scripts",
    "docs",
    "templates",
    "data/reference",
    "outputs/metrics",
)
_DATASET_DIRS = (
    "data/raw/bci_competition_iv_2a",
    "data/raw/bci_competition_iv_2a_labels",
    "data/raw/bci_competition_iv_2b",
    "data/raw/chowdhury_cse_uael",
)
_MAX_TEXT_BYTES = 1_000_000


@dataclass(frozen=True)
class SupportChunk:
    """One retrievable evidence unit."""

    source: str
    kind: str
    text: str
    start_line: int | None = None
    end_line: int | None = None
    page: int | None = None

    @property
    def locator(self) -> str:
        if self.page is not None:
            return f"{self.source} · page {self.page}"
        if self.start_line is not None and self.end_line is not None:
            return f"{self.source} · lines {self.start_line}-{self.end_line}"
        return self.source


@dataclass(frozen=True)
class RetrievalHit:
    """Retrieved chunk and cosine-like TF-IDF score."""

    chunk: SupportChunk
    score: float


@dataclass(frozen=True)
class SupportAnswer:
    """Answer returned to the Flask support page."""

    answer: str | None
    hits: tuple[RetrievalHit, ...]
    model: str
    model_error: str | None = None


class LocalSupportRAG:
    """Repository-grounded RAG using local retrieval and Ollama generation."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        proposal_path: str | Path | None = None,
        model: str = "llama3.2:1b",
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.proposal_path = (
            Path(proposal_path).expanduser().resolve()
            if proposal_path
            else None
        )
        self.model = str(model)
        self.ollama_url = str(ollama_url)
        self._chunks: list[SupportChunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._proposal_indexed = False
        self._dataset_inventory_count = 0

    @property
    def is_indexed(self) -> bool:
        return bool(self._chunks) and self._vectorizer is not None

    def build_index(self) -> None:
        """Build an in-memory local index from approved project sources."""

        chunks: list[SupportChunk] = []

        readme = self.project_root / "README.md"
        if readme.is_file():
            chunks.extend(self._chunks_from_text_file(readme))

        for relative_root in _SOURCE_ROOTS:
            root = self.project_root / relative_root
            if not root.exists():
                continue
            if root.is_file():
                chunks.extend(self._chunks_from_text_file(root))
                continue
            for path in sorted(root.rglob("*")):
                if (
                    path.is_file()
                    and path.suffix.lower() in _TEXT_SUFFIXES
                    and path.stat().st_size <= _MAX_TEXT_BYTES
                ):
                    chunks.extend(self._chunks_from_text_file(path))

        inventory_chunks = self._dataset_inventory_chunks()
        chunks.extend(inventory_chunks)
        self._dataset_inventory_count = len(inventory_chunks)

        proposal_chunks = self._proposal_chunks()
        chunks.extend(proposal_chunks)
        self._proposal_indexed = bool(proposal_chunks)

        if not chunks:
            raise RuntimeError("No support-layer source material could be indexed.")

        corpus = [chunk.text for chunk in chunks]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=60_000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(corpus)

        self._chunks = chunks
        self._vectorizer = vectorizer
        self._matrix = matrix

    def status(self) -> dict[str, object]:
        """Return index provenance for presentation on the support page."""

        if not self.is_indexed:
            self.build_index()

        kinds: dict[str, int] = {}
        for chunk in self._chunks:
            kinds[chunk.kind] = kinds.get(chunk.kind, 0) + 1

        return {
            "chunk_count": len(self._chunks),
            "kinds": kinds,
            "proposal_path": str(self.proposal_path) if self.proposal_path else None,
            "proposal_indexed": self._proposal_indexed,
            "dataset_inventory_count": self._dataset_inventory_count,
            "model": self.model,
            "ollama_url": self.ollama_url,
        }

    def retrieve(self, query: str, *, top_k: int = 6) -> tuple[RetrievalHit, ...]:
        """Return the most relevant local chunks for a question."""

        text = str(query).strip()
        if not text:
            return ()
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not self.is_indexed:
            self.build_index()

        assert self._vectorizer is not None
        query_vector = self._vectorizer.transform([text])
        scores = (self._matrix @ query_vector.T).toarray().reshape(-1)
        ranked = np.argsort(scores)[::-1]

        hits: list[RetrievalHit] = []
        for index in ranked:
            score = float(scores[index])
            if score <= 0.0:
                continue
            hits.append(RetrievalHit(self._chunks[int(index)], score))
            if len(hits) >= top_k:
                break
        return tuple(hits)

    def answer(
        self,
        query: str,
        *,
        top_k: int = 6,
        generator: Callable[[str], str] | None = None,
    ) -> SupportAnswer:
        """Retrieve evidence and ask the configured local Llama model."""

        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return SupportAnswer(
                answer=(
                    "No indexed project evidence matched this question. "
                    "Try naming the file, dataset, experiment, detector or route."
                ),
                hits=(),
                model=self.model,
            )

        prompt = self._build_prompt(query, hits)
        try:
            answer = generator(prompt) if generator else self._call_ollama(prompt)
            return SupportAnswer(
                answer=answer.strip(),
                hits=hits,
                model=self.model,
            )
        except (OSError, RuntimeError, ValueError, urlerror.URLError) as exc:
            return SupportAnswer(
                answer=None,
                hits=hits,
                model=self.model,
                model_error=str(exc),
            )

    def _build_prompt(
        self,
        query: str,
        hits: tuple[RetrievalHit, ...],
    ) -> str:
        context_parts = []
        for index, hit in enumerate(hits, start=1):
            context_parts.append(
                f"[S{index}] {hit.chunk.locator}\n{hit.chunk.text.strip()}"
            )
        context = "\n\n".join(context_parts)

        return (
            "You are the local research support layer for an auditable "
            "covariate-shift and adaptive-learning repository.\n"
            "Answer the user's question only from the retrieved context below.\n"
            "Rules:\n"
            "1. Distinguish implemented repository behaviour from proposal intentions.\n"
            "2. Distinguish BCI Competition benchmark EEG from procedurally generated "
            "synthetic streams.\n"
            "3. Never invent experiment results, file contents, dataset availability "
            "or protocol details.\n"
            "4. Cite supporting evidence inline with [S1], [S2], etc.\n"
            "5. If the retrieved context does not establish a claim, say so.\n"
            "6. Prefer concise technical answers with file paths and runnable commands "
            "when the context supports them.\n\n"
            f"Question:\n{query.strip()}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer:"
        )

    def _call_ollama(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            }
        ).encode("utf-8")
        req = urlrequest.Request(
            self.ollama_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}. Is {self.model!r} installed?"
            ) from exc
        except urlerror.URLError as exc:
            raise RuntimeError(
                "Could not reach local Ollama. Start Ollama and pull "
                f"{self.model!r} before asking the support model."
            ) from exc

        generated = body.get("response")
        if not isinstance(generated, str) or not generated.strip():
            raise RuntimeError("Ollama returned no generated response.")
        return generated

    def _chunks_from_text_file(self, path: Path) -> list[SupportChunk]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        if not text.strip():
            return []

        relative = self._relative_name(path)
        lines = text.splitlines()
        window = 70
        overlap = 12
        step = window - overlap
        chunks: list[SupportChunk] = []
        for start in range(0, len(lines), step):
            selected = lines[start : start + window]
            if not selected:
                break
            body = "\n".join(selected).strip()
            if body:
                chunks.append(
                    SupportChunk(
                        source=relative,
                        kind="repository",
                        text=body,
                        start_line=start + 1,
                        end_line=start + len(selected),
                    )
                )
            if start + window >= len(lines):
                break
        return chunks

    def _proposal_chunks(self) -> list[SupportChunk]:
        path = self.proposal_path
        if path is None or not path.is_file():
            return []

        if path.suffix.lower() == ".txt":
            return [
                SupportChunk(
                    source=path.name,
                    kind="proposal",
                    text=chunk.text,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                )
                for chunk in self._chunks_from_text_file(path)
            ]

        if path.suffix.lower() != ".pdf":
            return []

        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "Indexing the research-proposal PDF requires pypdf. "
                "Install requirements-support.txt."
            ) from exc

        chunks: list[SupportChunk] = []
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
            if not paragraphs:
                paragraphs = [text]
            buffer = ""
            for paragraph in paragraphs:
                candidate = f"{buffer}\n\n{paragraph}".strip()
                if buffer and len(candidate) > 4200:
                    chunks.append(
                        SupportChunk(
                            source=path.name,
                            kind="proposal",
                            text=buffer,
                            page=page_number,
                        )
                    )
                    buffer = paragraph
                else:
                    buffer = candidate
            if buffer:
                chunks.append(
                    SupportChunk(
                        source=path.name,
                        kind="proposal",
                        text=buffer,
                        page=page_number,
                    )
                )
        return chunks

    def _dataset_inventory_chunks(self) -> list[SupportChunk]:
        chunks: list[SupportChunk] = []
        for relative in _DATASET_DIRS:
            directory = self.project_root / relative
            if not directory.is_dir():
                continue
            files = sorted(path for path in directory.iterdir() if path.is_file())
            suffix_counts: dict[str, int] = {}
            for path in files:
                suffix = path.suffix.lower() or "<none>"
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
            examples = ", ".join(path.name for path in files[:40])
            summary = (
                f"Local dataset inventory for {relative}. "
                f"File count: {len(files)}. "
                f"Types: {suffix_counts}. "
                f"Files/sample: {examples or 'none'}."
            )
            chunks.append(
                SupportChunk(
                    source=relative,
                    kind="dataset-inventory",
                    text=summary,
                )
            )
        return chunks

    def _relative_name(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return str(path)
