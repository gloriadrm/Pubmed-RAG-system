"""
test_data_actualization.py
---------------------------
Tests pytest de pubmed_updater y oa_updater contra una colección Qdrant
temporal aislada — creada y destruida en cada test, nunca toca la colección
de producción (biomedical_rag). No dependen de red real hacia NLM/NCBI: los
update files, el CSV OA y las respuestas de PMC se sustituyen por
fixtures/mocks.

Requiere Qdrant accesible en QDRANT_URL (el mismo que usa la app).

Uso:
    pytest test/test_data_actualization.py -v
"""

import pytest
from qdrant_client.models import VectorParams, Distance, PointStruct

import src.ingest.indexer as ix
import src.ingest.pmc as pmc_module
import src.data_actualization.pubmed_updater as pu
import src.data_actualization.oa_updater as oa
from src.data_actualization.pmc_oa_client import OaMetadata, OaMetadataError

TEST_COLLECTION = "test_actualizations_pytest"


# -------------- Fixtures y helpers --------------

@pytest.fixture
def qdrant_collection(monkeypatch):
    """Colección Qdrant temporal y aislada. Redirige COLLECTION_NAME en los
    módulos bajo test para no tocar nunca la colección de producción."""
    client = ix.client
    if TEST_COLLECTION in [c.name for c in client.get_collections().collections]:
        client.delete_collection(TEST_COLLECTION)
    client.create_collection(
        TEST_COLLECTION,
        vectors_config=VectorParams(size=ix.VECTOR_SIZE, distance=Distance.COSINE),
    )

    # ix.COLLECTION_NAME cubre indexer.py y, por resolución perezosa en
    # tiempo de llamada, también oa_updater.py (que la reimporta dentro de
    # cada función). pubmed_updater.py la capturó como binding propio al
    # importar el módulo, así que necesita su propio patch.
    monkeypatch.setattr(ix, "COLLECTION_NAME", TEST_COLLECTION)
    monkeypatch.setattr(pu, "COLLECTION_NAME", TEST_COLLECTION)

    yield client

    client.delete_collection(TEST_COLLECTION)


def _vec():
    return [0.0] * ix.VECTOR_SIZE


def _put_pubmed(client, pm_id, **overrides):
    metadata = {
        "document_id": f"pubmed_{pm_id}", "source": "pubmed", "pm_id": pm_id,
        "pmc_id": None, "doi": None, "title": "Original title", "authors": [],
        "journal": "J", "language": "eng", "keywords": [], "pub_date": "2020-01-01",
        "date_revised": "2020-01-01", "license": None, "pmc_last_updated": None,
    }
    metadata.update(overrides)
    client.upsert(TEST_COLLECTION, points=[PointStruct(
        id=ix._make_id(f"pubmed_{pm_id}"), vector=_vec(),
        payload={"embedding_text": "x", "metadata": metadata},
    )])


def _put_pmc_chunk(client, pmc_id, chunk_index, total_chunks, **overrides):
    chunk_id = f"pmc_{pmc_id}_c{chunk_index}"
    metadata = {
        "document_id": chunk_id, "source": "pmc", "pmc_id": pmc_id, "pm_id": None,
        "doi": None, "title": "PMC title", "section": "Methods",
        "chunk_index": chunk_index, "total_chunks": total_chunks, "abstract": None,
        "authors": [], "journal": "J", "language": "eng", "keywords": [],
        "article_type": None, "pub_date": "2020-01-01", "license": "CC BY",
        "pmc_last_updated": "2024-01-01 00:00:00",
    }
    metadata.update(overrides)
    client.upsert(TEST_COLLECTION, points=[PointStruct(
        id=ix._make_id(chunk_id), vector=_vec(),
        payload={"embedding_text": "x", "metadata": metadata},
    )])


def _meta(client, doc_id):
    points = client.retrieve(TEST_COLLECTION, ids=[ix._make_id(doc_id)], with_payload=True)
    return points[0].payload["metadata"] if points else None


def _revised_article(pm_id, title="Revised title"):
    """Artículo tal como lo produce parse_update_file(): metadatos PubMed
    puros, nunca pmc_id/license/pmc_last_updated (eso solo lo añade
    pipeline.ingest_query tras cruzar con el OA CSV)."""
    return {
        "document_id": f"pubmed_{pm_id}", "source": "pubmed", "pm_id": pm_id, "doi": None,
        "title": title, "abstract": "revised abstract",
        "embedding_text": f"{title} revised abstract",
        "metadata": {"journal": "J", "language": "eng", "keywords": [], "authors": [],
                     "pub_date_raw": "2020", "article_date": None, "date_revised": "2026-07-19",
                     "pub_date": "2020-01-01"},
    }


def _oa_meta(pmcid, pmid=None, license_code="CC BY", is_retracted=False, last_modified="2025-06-01T00:00:00.000Z"):
    return OaMetadata(pmcid=pmcid, pmid=pmid, license_code=license_code,
                       is_retracted=is_retracted, last_modified=last_modified)


def _fake_pmc_chunk(pmc_id, chunk_index=0, total_chunks=1, title="Updated title", text="updated text"):
    return {
        "chunk_id": f"pmc_{pmc_id}_c{chunk_index}", "source": "pmc", "pmc_id": pmc_id,
        "pm_id": None, "doi": None, "title": title, "section": "Methods", "text": text,
        "chunk_index": chunk_index, "total_chunks": total_chunks,
        "embedding_text": f"{title} | Methods | {text}",
        "metadata": {"abstract": None, "authors": [], "journal": "J", "language": "eng",
                     "article_type": None, "keywords": [], "pub_date": "2020-01-01",
                     "pmc_last_updated": None, "license": None},
    }


# ============================================================
# PubMed updater
# ============================================================

class TestPubmedUpdater:

    def test_revision_preserves_pmc_enrichment(self, qdrant_collection):
        """1. Una revisión MEDLINE conserva pmc_id, license y pmc_last_updated."""
        client = qdrant_collection
        _put_pubmed(client, "111", pmc_id="PMC1111111", license="CC BY",
                    pmc_last_updated="2020-01-01 00:00:00")

        pu.apply_pubmed_updates(articles=[_revised_article("111")],
                                 deleted_pmids=[], indexed_pmids={"111"})

        md = _meta(client, "pubmed_111")
        assert md["title"] == "Revised title"          # sí se actualiza lo que trae la revisión
        assert md["pmc_id"] == "PMC1111111"             # pero se conserva el enriquecimiento
        assert md["license"] == "CC BY"
        assert md["pmc_last_updated"] == "2020-01-01 00:00:00"

    def test_delete_citation_removes_pmid(self, qdrant_collection):
        """2. DeleteCitation elimina el PMID."""
        client = qdrant_collection
        _put_pubmed(client, "222")

        pu.apply_pubmed_updates(articles=[], deleted_pmids=["222"], indexed_pmids={"222"})

        assert _meta(client, "pubmed_222") is None

    def test_deleted_pmid_does_not_resurrect_in_later_file_same_run(self, qdrant_collection):
        """3. Un fichero posterior en la misma ejecución no resucita ese PMID."""
        client = qdrant_collection
        _put_pubmed(client, "333")
        indexed_pmids = {"333"}

        # Fichero 1: DeleteCitation de 333
        deleted = pu.apply_pubmed_updates(articles=[], deleted_pmids=["333"], indexed_pmids=indexed_pmids)
        indexed_pmids -= deleted  # esto es lo que run_pubmed_update hace ahora tras cada fichero

        # Fichero 2 (misma ejecución, mismo indexed_pmids ya refrescado): revisión del mismo PMID
        pu.apply_pubmed_updates(articles=[_revised_article("333")], deleted_pmids=[], indexed_pmids=indexed_pmids)

        assert _meta(client, "pubmed_333") is None

    def test_non_indexed_pmid_is_ignored(self, qdrant_collection):
        """4. Un artículo nuevo no indexado se ignora."""
        client = qdrant_collection

        pu.apply_pubmed_updates(articles=[_revised_article("444")], deleted_pmids=[], indexed_pmids=set())

        assert _meta(client, "pubmed_444") is None


# ============================================================
# OA updater
# ============================================================

class TestOaUpdater:

    def test_only_indexed_pmc_ids_are_reprocessed(self, qdrant_collection, monkeypatch):
        """5. Solo se reprocesan PMC IDs ya indexados."""
        client = qdrant_collection
        _put_pmc_chunk(client, "PMC1000001", 0, 1)
        _put_pubmed(client, "1", pmc_id="PMC1000001", license="CC BY", pmc_last_updated="2024-01-01 00:00:00")
        # PMC2000002 nunca se indexa en Qdrant — no debe consultarse.

        queried = []

        def fake_fetch(pmc_id, version=1):
            queried.append(pmc_id)
            return _oa_meta(pmc_id)   # last_modified distinto de lo almacenado -> reingesta

        monkeypatch.setattr(oa, "fetch_oa_metadata", fake_fetch)
        monkeypatch.setattr(pmc_module, "fetch_pmc_full_text", lambda ids: "<xml/>")
        monkeypatch.setattr(pmc_module, "parse_pmc_xml_to_chunks",
                             lambda xml: [_fake_pmc_chunk("PMC1000001")])

        oa.run_daily_update()

        assert queried == ["PMC1000001"]   # PMC2000002 nunca se consultó: no está indexado
        updated = _meta(client, "pmc_PMC1000001_c0")
        assert updated["title"] == "Updated title"

    def test_change_of_non_indexed_pmc_is_ignored(self, qdrant_collection, monkeypatch):
        """6. Un cambio global de un PMC no indexado se ignora."""
        # Nada indexado en absoluto en esta colección.
        queried = []
        monkeypatch.setattr(oa, "fetch_oa_metadata", lambda pmcid, version=1: queried.append(pmcid) or _oa_meta(pmcid))

        oa.run_daily_update()

        assert queried == []   # sin pmc_ids indexados, no hay nada que consultar

    def test_removal_deletes_chunks_and_cleans_pubmed_enrichment(self, qdrant_collection, monkeypatch):
        """7. Una retirada elimina chunks PMC y limpia el enriquecimiento PubMed."""
        client = qdrant_collection
        _put_pmc_chunk(client, "PMC3000003", 0, 1)
        _put_pubmed(client, "3", pmc_id="PMC3000003", license="CC BY", pmc_last_updated="2024-01-01 00:00:00")

        # 404 confirmado: ya no está en el PMC Open Access Subset.
        monkeypatch.setattr(oa, "fetch_oa_metadata", lambda pmcid, version=1: None)

        oa.run_daily_update()

        assert _meta(client, "pmc_PMC3000003_c0") is None   # chunk PMC borrado

        pubmed_md = _meta(client, "pubmed_3")
        assert pubmed_md is not None                         # el documento PubMed se conserva
        assert pubmed_md["pmc_id"] is None
        assert pubmed_md["license"] is None
        assert pubmed_md["pmc_last_updated"] is None

    def test_retracted_article_is_treated_as_removal(self, qdrant_collection, monkeypatch):
        """Un artículo retractado se trata como retirada (distinto de licencia no permitida)."""
        client = qdrant_collection
        _put_pmc_chunk(client, "PMC7000007", 0, 1)
        _put_pubmed(client, "7", pmc_id="PMC7000007", license="CC BY", pmc_last_updated="2024-01-01 00:00:00")

        monkeypatch.setattr(oa, "fetch_oa_metadata",
                             lambda pmcid, version=1: _oa_meta(pmcid, is_retracted=True))

        oa.run_daily_update()

        assert _meta(client, "pmc_PMC7000007_c0") is None
        assert _meta(client, "pubmed_7")["pmc_id"] is None

    def test_license_change_to_disallowed_is_treated_as_removal(self, qdrant_collection, monkeypatch):
        """8. Un cambio a licencia no permitida se trata como retirada."""
        client = qdrant_collection
        _put_pmc_chunk(client, "PMC4000004", 0, 1)
        _put_pubmed(client, "4", pmc_id="PMC4000004", license="CC BY", pmc_last_updated="2024-01-01 00:00:00")

        fetched = []
        monkeypatch.setattr(oa, "fetch_oa_metadata",
                             lambda pmcid, version=1: _oa_meta(pmcid, license_code="CC BY-ND"))
        monkeypatch.setattr(pmc_module, "fetch_pmc_full_text",
                             lambda ids: (fetched.extend(ids), "<xml/>")[1])

        oa.run_daily_update()

        assert _meta(client, "pmc_PMC4000004_c0") is None
        assert fetched == []   # nunca se reingesta: se trata como baja, no como cambio
        pubmed_md = _meta(client, "pubmed_4")
        assert pubmed_md["license"] is None

    def test_failed_reingest_keeps_previous_chunks(self, qdrant_collection, monkeypatch):
        """9. Una reingesta fallida no elimina los chunks anteriores."""
        client = qdrant_collection
        _put_pmc_chunk(client, "PMC5000005", 0, 1, title="Original title")
        _put_pubmed(client, "5", pmc_id="PMC5000005", license="CC BY", pmc_last_updated="2024-01-01 00:00:00")

        def fake_fetch(pmc_ids):
            raise ConnectionError("NCBI no disponible")

        monkeypatch.setattr(pmc_module, "fetch_pmc_full_text", fake_fetch)

        to_reingest = {"PMC5000005": _oa_meta("PMC5000005")}
        oa.reingest_pmc_articles(to_reingest)   # no debe propagar la excepción — se registra y continúa

        md = _meta(client, "pmc_PMC5000005_c0")
        assert md is not None
        assert md["title"] == "Original title"   # intacto, no se ha tocado

    def test_network_failure_checking_article_leaves_it_untouched(self, qdrant_collection, monkeypatch):
        """Un fallo de red al comprobar un pmc_id indexado no lo trata como
        retirada ni como cambio — se dejaba como está (nunca se equipara un
        fallo transitorio con un 404 confirmado)."""
        client = qdrant_collection
        _put_pmc_chunk(client, "PMC8000008", 0, 1, title="Untouched title")
        _put_pubmed(client, "8", pmc_id="PMC8000008", license="CC BY", pmc_last_updated="2024-01-01 00:00:00")

        def fake_fetch(pmcid, version=1):
            raise OaMetadataError("timeout consultando metadata")

        monkeypatch.setattr(oa, "fetch_oa_metadata", fake_fetch)

        oa.run_daily_update()

        md = _meta(client, "pmc_PMC8000008_c0")
        assert md is not None
        assert md["title"] == "Untouched title"
        pubmed_md = _meta(client, "pubmed_8")
        assert pubmed_md["pmc_id"] == "PMC8000008"   # el enriquecimiento tampoco se tocó

    def test_unchanged_article_is_not_reingested(self, qdrant_collection, monkeypatch):
        """Si last_modified coincide con lo almacenado, no hay nada que reingerir."""
        client = qdrant_collection
        _put_pmc_chunk(client, "PMC9000009", 0, 1, title="Stable title",
                        pmc_last_updated="2024-01-01 00:00:00")
        _put_pubmed(client, "9", pmc_id="PMC9000009", license="CC BY", pmc_last_updated="2024-01-01 00:00:00")

        fetched = []
        monkeypatch.setattr(oa, "fetch_oa_metadata",
                             lambda pmcid, version=1: _oa_meta(pmcid, last_modified="2024-01-01 00:00:00"))
        monkeypatch.setattr(pmc_module, "fetch_pmc_full_text",
                             lambda ids: (fetched.extend(ids), "<xml/>")[1])

        oa.run_daily_update()

        assert fetched == []   # nada cambió -> no se reingesta
        assert _meta(client, "pmc_PMC9000009_c0")["title"] == "Stable title"
