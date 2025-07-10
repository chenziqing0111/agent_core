from agent_core.agents.literature_agent.prompts import (
    build_disease_prompt,
    build_treatment_prompt,
    build_target_prompt
)
from agent_core.clients.llm_client import call_llm
from agent_core.agents.literature_agent.retriever.pubmed_retriever import get_pubmed_abstracts


def analyze_disease_mechanism(gene: str, abstracts: list[str]) -> str:
    prompt = build_disease_prompt(gene, abstracts)
    return call_llm(prompt)


def analyze_treatment_strategy(gene: str, abstracts: list[str]) -> str:
    prompt = build_treatment_prompt(gene, abstracts)
    return call_llm(prompt)


def analyze_target_info(gene: str, abstracts: list[str]) -> str:
    prompt = build_target_prompt(gene, abstracts)
    return call_llm(prompt)


def run_literature_agent(gene: str) -> dict:
    print(f"[LiteratureAgent] 正在检索 {gene} 的 PubMed 文献摘要...")
    docs = get_pubmed_abstracts(gene, retmax=100)
    # abstracts = [doc["Abstract"] for doc in docs if doc.get("Abstract")]
    numbered_docs = [
        {
            "Index": i + 1,
            "PMID": doc["PMID"],
            "Title": doc["Title"],
            "Abstract": doc["Abstract"]
        }
        for i, doc in enumerate(docs) if doc.get("Abstract")
    ]

    numbered_abstracts = [f"[{d['Index']}] {d['Abstract']}" for d in numbered_docs]
    abstracts_block = "\n\n".join(numbered_abstracts)
    print("[LiteratureAgent] 启动 LLM 分析任务：三项并行分析...")
    disease_result = analyze_disease_mechanism(gene, abstracts_block)
    treatment_result = analyze_treatment_strategy(gene, abstracts_block)
    target_result = analyze_target_info(gene, abstracts_block)

    references = [
        {"PMID": d["PMID"], "Title": d["Title"], "Index": d["Index"]}
        for d in numbered_docs
    ]

    print("[LiteratureAgent] 分析完成，返回结构化结果...")
    return {
        "gene": gene,
        "disease_mechanism": disease_result,
        "treatment_strategy": treatment_result,
        "target_analysis": target_result,
        "references": references
    }

