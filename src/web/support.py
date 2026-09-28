"""Flask page for the local repository RAG support layer."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Blueprint, current_app, redirect, render_template, request, url_for

from src.support.rag import LocalSupportRAG


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUPPORT_PROPOSAL_PATH = (
    PROJECT_ROOT / "data" / "reference" / "Research Proposal Haocheng Lin(5).pdf"
)
DEFAULT_SUPPORT_MODEL = "llama3.2:1b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

support_bp = Blueprint("support", __name__)

_cached_rag: LocalSupportRAG | None = None
_cached_signature: tuple[str, str, str] | None = None


def configure_support_defaults(app) -> None:
    """Register local-only support-layer defaults on the Flask app."""

    app.config.setdefault(
        "SUPPORT_RAG_PROPOSAL_PATH",
        os.environ.get(
            "SUPPORT_RAG_PROPOSAL_PATH",
            str(DEFAULT_SUPPORT_PROPOSAL_PATH),
        ),
    )
    app.config.setdefault(
        "SUPPORT_RAG_MODEL",
        os.environ.get("SUPPORT_RAG_MODEL", DEFAULT_SUPPORT_MODEL),
    )
    app.config.setdefault(
        "SUPPORT_RAG_OLLAMA_URL",
        os.environ.get("SUPPORT_RAG_OLLAMA_URL", DEFAULT_OLLAMA_URL),
    )


def clear_support_cache() -> None:
    global _cached_rag, _cached_signature
    _cached_rag = None
    _cached_signature = None


def _support_rag() -> LocalSupportRAG:
    global _cached_rag, _cached_signature

    proposal = str(current_app.config.get("SUPPORT_RAG_PROPOSAL_PATH", ""))
    model = str(current_app.config.get("SUPPORT_RAG_MODEL", DEFAULT_SUPPORT_MODEL))
    ollama_url = str(
        current_app.config.get("SUPPORT_RAG_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    )
    signature = (proposal, model, ollama_url)

    if _cached_rag is None or _cached_signature != signature:
        _cached_rag = LocalSupportRAG(
            PROJECT_ROOT,
            proposal_path=proposal or None,
            model=model,
            ollama_url=ollama_url,
        )
        _cached_rag.build_index()
        _cached_signature = signature
    return _cached_rag


@support_bp.route("/support", methods=["GET", "POST"])
def support_layer():
    """Ask the local Llama support model about code, datasets and proposal scope."""

    query = ""
    result = None
    page_error = None
    status = None

    try:
        rag = _support_rag()
        status = rag.status()
        if request.method == "POST":
            query = request.form.get("query", "").strip()
            if query:
                result = rag.answer(query)
    except (OSError, RuntimeError, ValueError) as exc:
        page_error = str(exc)

    return render_template(
        "support.html",
        query=query,
        result=result,
        status=status,
        page_error=page_error,
    )


@support_bp.post("/support/reindex")
def reindex_support_layer():
    clear_support_cache()
    return redirect(url_for("support.support_layer"))
