from typing import List
import re
import time
from agent_core.agents.clinical_agent.retriever.ctg_retriever import (
    get_trials_basic,
    format_brief_trials_markdown
)
from agent_core.agents.clinical_agent.prompts import (
    build_brief_selection_prompt,
    build_detailed_analysis_prompt,
)
from agent_core.clients.llm_client import call_llm
# ---------------------------------------------------------------------------
# FIRST‑ROUND: 让 LLM 总结所有 brief，并返回 Top‑N NCT IDs
# ---------------------------------------------------------------------------

def _select_top_n(gene: str, trials: List[dict], top_n: int = 3) -> List[str]:
    """Combine brief summaries → ask LLM → extract Top‑N NCT IDs"""
    # 组装文本块（限制摘要 1000 字/条）
    blocks = []
    for t in trials:
        ident = t["protocolSection"]["identificationModule"]
        desc = t["protocolSection"].get("descriptionModule", {})
        nct  = ident.get("nctId", "")
        title = ident.get("briefTitle", "")
        brief_raw = desc.get("briefSummary", {})
        brief = brief_raw.get("textBlock") if isinstance(brief_raw, dict) else str(brief_raw)
        blocks.append(f"[{nct}] {title}\n{brief[:1000]}")

    combined = "\n\n".join(blocks)
    prompt = build_brief_selection_prompt(gene, combined, top_n)
    reply = call_llm(prompt)

    # 抓取 NCT IDs
    nct_ids = re.findall(r"NCT\d{8}", reply)
    return nct_ids[:top_n]


# ---------------------------------------------------------------------------
# SECOND‑ROUND: 对 Top‑N 进行详细分析
# ---------------------------------------------------------------------------

def _deep_analysis(gene: str, selected_trials: List[dict]) -> str:
    sections = []
    for t in selected_trials:
        ident = t["protocolSection"]["identificationModule"]
        desc = t["protocolSection"].get("descriptionModule", {})
        nct  = ident.get("nctId", "")
        title = ident.get("briefTitle", "")
        detail_raw = desc.get("detailedDescription", {})
        detail = detail_raw.get("textBlock") if isinstance(detail_raw, dict) else str(detail_raw)

        prompt = build_detailed_analysis_prompt(gene, title, detail[:4000])  # 截断防溢出
        analysis_md = call_llm(prompt)
        sections.append(f"### {title} ({nct})\n\n" + analysis_md)
        time.sleep(0.5)  # 轻微 pause，避免速率限制

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# PUBLIC ENTRY
# ---------------------------------------------------------------------------

def run_clinical_agent(gene: str, *, max_trials: int = 40, top_n: int = 3) -> str:
    print(f"[ClinicalAgent] 拉取 {gene} 相关试验……")
    trials = get_trials_basic(gene, page_size=20, max_pages=max_trials // 20 + 1)

    # 生成简要队列 markdown
    brief_md = format_brief_trials_markdown(trials)

    # LLM 选出 Top‑N
    top_ids = _select_top_n(gene, trials, top_n=top_n)
    print(f"[ClinicalAgent] LLM 选出 Top‑{top_n}：", top_ids)

    # 取对应 trial 对象
    selected_trials = [t for t in trials if t["protocolSection"]["identificationModule"].get("nctId") in top_ids]

    # 深度分析
    deep_md = _deep_analysis(gene, selected_trials)

    full_report = (
        "## 试验概览 (Brief Summary)\n\n" + brief_md +
        f"\n\n## 深度分析 Top‑{top_n}\n\n" + deep_md
    )

    return full_report


if __name__ == "__main__":
    md_report = run_clinical_agent("PCSK9", max_trials=40, top_n=3)
    print(md_report)
