from typing import List, Dict
import xml.etree.ElementTree as ET
from Bio import Entrez   # pip install biopython

Entrez.email   = "czqrainy@gmail.com"
Entrez.api_key = "983222f9d5a2a81facd7d158791d933e6408"   # 没有就删掉这一行

def get_pubmed_abstracts(query: str, retmax: int = 20) -> List[Dict]:
    """返回 Title / Abstract / PMID 的列表"""
    # ① esearch：拿 PMID 列表
    ids = Entrez.read(
        Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    )["IdList"]
    if not ids:
        return []

    # ② efetch：一次性把摘要抓下来（XML）
    root = ET.fromstring(
        Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml").read()
    )

    records = []
    for art in root.findall(".//PubmedArticle"):
        pmid     = art.findtext(".//PMID")
        title    = art.findtext(".//ArticleTitle", default="No title")
        abstract = art.findtext(".//AbstractText",  default="No abstract")
        records.append({"PMID": pmid, "Title": title, "Abstract": abstract})
    return records
