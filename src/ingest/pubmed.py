import os
from dotenv import load_dotenv
import requests
from xml.etree import ElementTree as ET

load_dotenv()
ncbi_api_key = os.getenv('NCBI_API_KEY')


# -------------- ESTRAER IDS + ABSTRACT + METADATA --------------

def search_pubmed_ids(
    terms: str,
    n: int,
    medline_only: bool = True,
    require_abstract: bool = True,
    use_history: bool = False
):
    """
    Busca artículos en PubMed con una query.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = base_url + "esearch.fcgi"

    filters = []

    # Filtramos publicaciones con status = medline
    if medline_only:
        filters.append("medline[sb]")

    # Obligatorio que tenga abstract
    if require_abstract:
        filters.append("hasabstract")

    # Construcción final del término
    if filters:
        final_term = f"({terms}) AND " + " AND ".join(filters)
    else:
        final_term = terms

    params_search = {
        "db": "pubmed",
        "term": final_term,
        "retmax": n,  # número máximo de IDs a devolver
        "retmode": "json", # formato de salida
        "api_key": ncbi_api_key,
    }

    if use_history:
        params_search["usehistory"] = "y"

    response = requests.get(search_url, params=params_search, timeout=60)
    response.raise_for_status()
    data = response.json()

    esearch_result = data.get("esearchresult", {})

    result = {
        "pmids": esearch_result.get("idlist", []),
        "translationset": esearch_result.get("translationset", []),
        "count": esearch_result.get("count"),
        "query_used": final_term  # 👈 MUY útil para debug
    }

    if use_history:
        result["webenv"] = esearch_result.get("webenv")
        result["query_key"] = esearch_result.get("querykey")

    return result

def fetch_pubmed_data(ids: list[str]):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    fetch_url = base_url + "efetch.fcgi"

    params_fetch = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
        "api_key": ncbi_api_key
    }

    response = requests.get(fetch_url, params=params_fetch, timeout=60)
    response.raise_for_status()
    return response.text



# -------------- PARSER HELPERS --------------

def get_full_text(node, path):
    elem = node.find(path)
    if elem is None:
        return None

    text = "".join(elem.itertext()).strip()
    return text if text else None

def get_all_full_texts(node, path):
    elems = node.findall(path)
    values = []

    for elem in elems:
        text = "".join(elem.itertext()).strip()
        if text:
            values.append(text)

    return values

def get_full_text(elem, path, default=None):
    node = elem.find(path)
    if node is not None and node.text:
        return node.text.strip()
    return default


def get_all_full_texts(elem, path):
    nodes = elem.findall(path)
    values = []
    for node in nodes:
        if node.text and node.text.strip():
            values.append(node.text.strip())
    return values


def parse_author_list(article_elem):
    authors = []
    author_nodes = article_elem.findall(".//AuthorList/Author")

    for author in author_nodes:
        last_name = get_full_text(author, "LastName")
        name = get_full_text(author, "ForeName")
        initials = get_full_text(author, "Initials")

        collective_name = get_full_text(author, "CollectiveName")

        if collective_name:
            authors.append({
                "collective_name": collective_name
            })
        else:
            authors.append({
                "last_name": last_name,
                "name": name,
                "initials": initials
            })

    return authors


def parse_keyword_list(article_elem):
    keywords = []
    keyword_nodes = article_elem.findall(".//KeywordList/Keyword")

    for kw in keyword_nodes:
        if kw.text and kw.text.strip():
            keywords.append(kw.text.strip())

    return keywords


def parse_article_ids(article_elem):
    article_ids = {
        "pm_id": None,
        "doi": None
    }

    # PMID principal
    pmid = get_full_text(article_elem, ".//MedlineCitation/PMID")
    article_ids["pm_id"] = pmid

    # DOI dentro de ArticleIdList
    for article_id in article_elem.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = article_id.attrib.get("IdType")
        if id_type == "doi" and article_id.text:
            article_ids["doi"] = article_id.text.strip()

    return article_ids


def parse_pub_date(article_elem):
    """
    Devuelve:
    - pub_date_raw: representación simple de PubDate
    - article_date: fecha completa si existe
    """
    pub_date_node = article_elem.find(".//Article/Journal/JournalIssue/PubDate")

    year = month = day = None
    medline_date = None

    if pub_date_node is not None:
        year = get_full_text(pub_date_node, "Year")
        month = get_full_text(pub_date_node, "Month")
        day = get_full_text(pub_date_node, "Day")
        medline_date = get_full_text(pub_date_node, "MedlineDate")

    if year:
        pub_date_raw = "-".join([x for x in [year, month, day] if x])
    else:
        pub_date_raw = medline_date

    article_date = None
    article_date_node = article_elem.find(".//Article/ArticleDate")
    if article_date_node is not None:
        a_year = get_full_text(article_date_node, "Year")
        a_month = get_full_text(article_date_node, "Month")
        a_day = get_full_text(article_date_node, "Day")

        if a_year and a_month and a_day:
            article_date = f"{a_year}-{a_month.zfill(2)}-{a_day.zfill(2)}"

    return pub_date_raw, article_date


def parse_date_revised(article_elem):
    date_node = article_elem.find(".//DateRevised")
    if date_node is None:
        return None

    year = get_full_text(date_node, "Year")
    month = get_full_text(date_node, "Month")
    day = get_full_text(date_node, "Day")

    if year and month and day:
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return None


def parse_abstract(article_elem):
    abstract_nodes = article_elem.findall(".//Article/Abstract/AbstractText")

    parts = []
    for node in abstract_nodes:
        label = node.attrib.get("Label")
        text = "".join(node.itertext()).strip()

        if text:
            if label:
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)

    return " ".join(parts) if parts else None


def choose_best_date(pub_date_raw, article_date):
    return article_date if article_date else pub_date_raw


# -------------- PARSE XML --------------

def parse_pubmed_article_fields(article_elem):
    article_ids = parse_article_ids(article_elem)

    pmid = article_ids["pm_id"]
    doi = article_ids["doi"]

    title = get_full_text(article_elem, ".//Article/ArticleTitle")
    abstract = parse_abstract(article_elem)
    journal = get_full_text(article_elem, ".//Article/Journal/Title")

    language_list = get_all_full_texts(article_elem, ".//Article/Language")
    language = language_list[0] if language_list else None

    keywords = parse_keyword_list(article_elem)
    authors = parse_author_list(article_elem)

    pub_date_raw, article_date = parse_pub_date(article_elem)
    date_revised = parse_date_revised(article_elem)
    best_date = choose_best_date(pub_date_raw, article_date)

    return {
        "pm_id": pmid,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "language": language,
        "keywords": keywords,
        "authors": authors,
        "pub_date_raw": pub_date_raw,
        "article_date": article_date,
        "date_revised": date_revised,
        "pub_date": best_date
    }



# -------------- BUILD FINAL DICT --------------

def build_pubmed_article_dict(fields):
    embedding_text = " ".join(
        part for part in [fields["title"], fields["abstract"]] if part
    ).strip()

    return {
        "document_id": f"pubmed_{fields['pm_id']}",
        "source": "pubmed",

        "pm_id": fields["pm_id"],
        "doi": fields["doi"],

        "title": fields["title"],
        "abstract": fields["abstract"],
        "embedding_text": embedding_text,

        "metadata": {
            "journal": fields["journal"],
            "language": fields["language"],
            "keywords": fields["keywords"],
            "authors": fields["authors"],
            "pub_date_raw": fields["pub_date_raw"],
            "article_date": fields["article_date"],
            "date_revised": fields["date_revised"],
            "pub_date": fields["pub_date"]
        }
    }

# -------------- 1 XML CONTAINS N ARTICLES --------------
def parse_pubmed_xml_to_dicts(xml_text: str):
    # Convierte el XML en un árbol
    root = ET.fromstring(xml_text)
    articles = []

    # Busca todos los artículos dentro del XML
    for article_elem in root.findall(".//PubmedArticle"):
        fields = parse_pubmed_article_fields(article_elem)
        article_dict = build_pubmed_article_dict(fields)
        articles.append(article_dict)

    return articles


# -------------- PIPELINE --------------

def build_pubmed_article_dicts(query: str, n: int, medline_only: bool = True):
    results = search_pubmed_ids(query, n, medline_only=medline_only)

    pmids = results["pmids"]
    translationset = results["translationset"]

    if not pmids:
        return {
            "query": query,
            "translationset": translationset,
            "pmids": [],
            "articles": []
        }

    xml_text = fetch_pubmed_data(pmids)
    articles = parse_pubmed_xml_to_dicts(xml_text)

    return {
        "query": query,
        "translationset": translationset,
        "pmids": pmids,
        "articles": articles
    }


