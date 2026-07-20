"""
test_pmc_oa_client.py
----------------------
Tests pytest de src/data_actualization/pmc_oa_client.py — el cliente que
sustituye a oa_file_list.csv (retirado por NCBI en 2026).

Todo mockeado con unittest.mock: no depende de red real ni de Qdrant.

Uso:
    pytest test/test_pmc_oa_client.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data_actualization.pmc_oa_client import (
    resolve_pmcids,
    fetch_oa_metadata,
    fetch_oa_metadata_bulk,
    OaMetadata,
    OaMetadataError,
)


# -------------- 1. Resolución PMID → PMCID --------------

class TestResolvePmcids:
    def _df(self):
        return pd.DataFrame({
            "pmc_id": ["PMC1000001", "PMC1000002", "PMC1000003"],
            "pm_id":  ["111",        "222",        "333"],
        })

    def test_resolves_known_pmids(self):
        result = resolve_pmcids(["111", "333"], self._df())
        assert result == {"111": "PMC1000001", "333": "PMC1000003"}

    def test_ignores_unknown_pmids(self):
        result = resolve_pmcids(["999"], self._df())
        assert result == {}

    def test_empty_pmid_list_returns_empty(self):
        result = resolve_pmcids([], self._df())
        assert result == {}

    def test_mixed_known_and_unknown(self):
        result = resolve_pmcids(["222", "999"], self._df())
        assert result == {"222": "PMC1000002"}


# -------------- 2. Metadata OA por artículo --------------

def _mock_response(status_code=200, json_body=None, raw_text=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    if raw_text is not None:
        resp.json.side_effect = ValueError("Expecting value")
    else:
        resp.json.return_value = json_body
    def raise_for_status():
        if status_code >= 400:
            raise requests.exceptions.HTTPError(f"{status_code} error")
    resp.raise_for_status.side_effect = raise_for_status
    return resp


VALID_JSON = {
    "pmcid": "PMC12855588",
    "version": 1,
    "pmid": 41623473,
    "doi": "10.1016/j.isci.2025.114581",
    "title": "Sample article",
    "is_pmc_openaccess": True,
    "is_manuscript": False,
    "is_historical_ocr": False,
    "is_retracted": False,
    "license_code": "CC BY",
    "xml_url": "s3://pmc-oa-opendata/PMC12855588.1/PMC12855588.1.xml",
}


class TestFetchOaMetadata:

    @patch("src.data_actualization.pmc_oa_client.requests.get")
    def test_reads_valid_json(self, mock_get):
        mock_get.return_value = _mock_response(
            200, VALID_JSON, headers={"Last-Modified": "Wed, 15 Jul 2026 10:00:00 GMT"}
        )
        meta = fetch_oa_metadata("PMC12855588")
        assert meta == OaMetadata(
            pmcid="PMC12855588",
            pmid="41623473",
            license_code="CC BY",
            is_retracted=False,
            last_modified="Wed, 15 Jul 2026 10:00:00 GMT",
        )

    @patch("src.data_actualization.pmc_oa_client.requests.get")
    def test_not_found_returns_none(self, mock_get):
        mock_get.return_value = _mock_response(404)
        assert fetch_oa_metadata("PMC00000000") is None

    @patch("src.data_actualization.pmc_oa_client.requests.get")
    def test_retracted_article(self, mock_get):
        body = {**VALID_JSON, "is_retracted": True}
        mock_get.return_value = _mock_response(200, body)
        meta = fetch_oa_metadata("PMC12855588")
        assert meta.is_retracted is True

    @patch("src.data_actualization.pmc_oa_client.requests.get")
    def test_license_code_null_is_valid_not_incomplete(self, mock_get):
        # license_code: null es un valor legítimo del JSON (artículo sin
        # licencia CC machine-readable) — no debe tratarse como respuesta
        # incompleta, la CLAVE está presente, solo su valor es None.
        body = {**VALID_JSON, "license_code": None}
        mock_get.return_value = _mock_response(200, body)
        meta = fetch_oa_metadata("PMC12855588")
        assert meta.license_code is None
        assert meta.is_retracted is False

    @patch("src.data_actualization.pmc_oa_client.requests.get")
    def test_network_failure_raises_not_none(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("connection refused")
        with pytest.raises(OaMetadataError):
            fetch_oa_metadata("PMC12855588")

    @patch("src.data_actualization.pmc_oa_client.requests.get")
    def test_timeout_raises_oametadataerror(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(OaMetadataError):
            fetch_oa_metadata("PMC12855588")

    @patch("src.data_actualization.pmc_oa_client.requests.get")
    def test_unexpected_status_raises(self, mock_get):
        mock_get.return_value = _mock_response(500)
        with pytest.raises(OaMetadataError):
            fetch_oa_metadata("PMC12855588")

    @patch("src.data_actualization.pmc_oa_client.requests.get")
    def test_malformed_json_raises(self, mock_get):
        mock_get.return_value = _mock_response(200, raw_text="not json{{{")
        with pytest.raises(OaMetadataError):
            fetch_oa_metadata("PMC12855588")

    @patch("src.data_actualization.pmc_oa_client.requests.get")
    def test_incomplete_response_missing_keys_raises(self, mock_get):
        # Respuesta truncada/incompleta: falta license_code e is_retracted.
        incomplete = {"pmcid": "PMC12855588", "pmid": 41623473}
        mock_get.return_value = _mock_response(200, incomplete)
        with pytest.raises(OaMetadataError):
            fetch_oa_metadata("PMC12855588")

    @patch("src.data_actualization.pmc_oa_client.requests.get")
    def test_pmid_null_maps_to_none(self, mock_get):
        body = {**VALID_JSON, "pmid": None}
        mock_get.return_value = _mock_response(200, body)
        meta = fetch_oa_metadata("PMC12855588")
        assert meta.pmid is None


class TestFetchOaMetadataBulk:

    @patch("src.data_actualization.pmc_oa_client.requests.get")
    def test_skips_not_found_and_failures_keeps_successes(self, mock_get):
        def side_effect(url, timeout=None):
            if "PMC1" in url:
                return _mock_response(200, {**VALID_JSON, "pmcid": "PMC1"})
            if "PMC2" in url:
                return _mock_response(404)
            raise requests.exceptions.ConnectionError("down")

        mock_get.side_effect = side_effect
        result = fetch_oa_metadata_bulk(["PMC1", "PMC2", "PMC3"])

        assert set(result.keys()) == {"PMC1"}
        assert result["PMC1"].pmcid == "PMC1"
