"""
structures.py
-------------
Modelos y tipos compartidos del pipeline RAG.

  SourceSelection   → salida estructurada del clasificador LLM
  RetrievalResult    → salida del retrieval real (puede diferir de SourceSelection
                       si hubo degradación, un pmc_id forzado por el request, etc.)
  normalize_pmc_id  → validación/normalización de PMC IDs, compartida entre la
                       frontera de la API (QueryRequest) y el pipeline interno

Casos de uso (QueryType):
  thematic_summary  → busca en abstracts PubMed (source=pubmed)
  specific_query    → busca en texto completo de UN artículo concreto (source=pmc + pmc_id)
  transversal_query → busca en texto completo de TODOS los artículos (source=pmc)
  none              → pregunta fuera del dominio biomédico o de las capacidades del sistema
"""

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.documents import Document

QueryType = Literal["thematic_summary", "specific_query", "transversal_query", "none"]


def normalize_pmc_id(value: str) -> str:
    """
    Normaliza un PMC ID a formato 'PMC' + dígitos. Acepta variantes con o sin
    prefijo 'PMC' (en cualquier capitalización) y espacios circundantes.

    Lanza ValueError si el resultado no matchea 'PMC' + dígitos — usado como
    field_validator en QueryRequest para que FastAPI devuelva un 422 claro
    ante un formato inválido, en vez de degradar silenciosamente.
    """
    normalized = value.strip().upper()
    if normalized.isdigit():
        normalized = f"PMC{normalized}"
    if not re.fullmatch(r"PMC\d+", normalized):
        raise ValueError(
            "El PMC ID debe tener el formato PMC seguido de dígitos, por ejemplo PMC12345678."
        )
    return normalized


class SourceSelection(BaseModel):
    query_type: QueryType = Field(
        ...,
        description=(
            "Classify the user's question into exactly one of the following query types:\n\n"

            "1. 'thematic_summary': use this when the user asks for a broad overview, summary, "
            "state of the art, recent papers, or general evidence about a biomedical topic. "
            "These queries are answered by retrieving multiple PubMed article-level documents "
            "(title + abstract). Examples: 'Give me the latest papers on gut microbiota and inflammation', "
            "'Summarize the evidence on probiotics and intestinal permeability'.\n\n"

            "2. 'specific_query': use this when the user is asking about one specific paper and wants "
            "details such as methodology, results, conclusions, limitations, sample size, findings... "
            "This applies when the question mentions a specific PMC ID, a paper title, or an author "
            "in a way that refers to one concrete study. These queries should be answered by searching "
            "within the full-text chunks of one single article.\n\n"

            "3. 'transversal_query': use this when the user is asking about a specific concept, mechanism, "
            "intervention, microorganism, or result across multiple papers at the full-text level. "
            "The goal is not to summarize abstracts broadly, but to retrieve detailed evidence from "
            "sections of several articles. Examples: 'What do studies say about Lactobacillus reuteri "
            "in inflammation?', 'Which papers report changes in intestinal permeability after probiotic use?'.\n\n"
            "4. 'none': use this when the question is outside the biomedical domain, or clearly outside "
            "the capabilities of this system (a biomedical literature RAG assistant)."
        )
    )

    pmc_id: Optional[str] = Field(
        default=None,
        description=(
            "Only for query_type='specific_query'. Extract the PMC ID if the user explicitly mentions it. "
            "The expected format is 'PMC' followed by digits, for example 'PMC12345678'. "
            "Return null if no PMC ID is explicitly mentioned."
        )
    )

    author_last_name: Optional[str] = Field(
        default=None,
        description=(
            "Only for query_type='specific_query'. Extract the last name of the author if the user refers "
            "to a specific paper by author name. Return null if no author last name is mentioned."
        )
    )

    author_name: Optional[str] = Field(
        default=None,
        description=(
            "Only for query_type='specific_query'. Extract the first name or given name of the author "
            "if explicitly mentioned. Return null if it is not mentioned or unknown."
        )
    )

    paper_title: Optional[str] = Field(
        default=None,
        description=(
            "Only for query_type='specific_query'. If the user explicitly quotes or mentions a paper title "
            "(e.g. 'In the paper X...', 'the study titled Y...'), extract it verbatim. "
            "Return null if no paper title is mentioned."
        )
    )

    reason: str = Field(
        ...,
        description=(
            "Provide a short explanation of why this query_type was selected. "
            "Mention the key signal used for classification, such as broad topic summary, "
            "specific paper reference, or cross-paper full-text evidence request."
        )
    )


@dataclass
class RetrievalResult:
    """
    Resultado real del retrieval — puede diferir de SourceSelection.query_type
    cuando el request trae un pmc_id explícito (que tiene prioridad absoluta,
    incluso sobre query_type='none') o cuando specific_query degrada a
    transversal_query por no poder resolver un artículo único.
    """
    docs: list[Document] = field(default_factory=list)
    retrieval_strategy: QueryType = "none"
    resolved_pmc_id: Optional[str] = None
    retrieval_note: Optional[str] = None
