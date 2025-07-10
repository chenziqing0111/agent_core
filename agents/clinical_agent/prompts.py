import textwrap

def build_brief_selection_prompt(gene: str, trials_block: str, top_n: int = 3) -> str:
    """Prompt to summarise progress & pick top‑N NCT IDs."""
    return textwrap.dedent(f"""
    你是一名资深临床研发分析师，任务是：
    1. 阅读下列关于 **{gene}** 的临床试验标题与简要摘要（briefSummary）。
    2. 先整体概括目前围绕 {gene} 的研发进展（1‑2 句话）。
    3. 选择最具价值、最直接与 {gene} 靶向/调控相关的 **{top_n}** 个研究队列，用于深入分析。

    ### 输出格式
    ```markdown
    **总体进展概括**：<一句话总结>

    **推荐 Top‑{top_n} NCT IDs**：NCT00000000, NCT11111111, NCT22222222
    ```

    ### 候选试验列表
    {trials_block}
    """)


def build_detailed_analysis_prompt(gene: str, title: str, detail: str) -> str:
    """Prompt to extract deep info from detailedDescription."""
    return textwrap.dedent(f"""
    请针对以下临床试验，围绕靶点 **{gene}** 进行深入信息提取，输出 Markdown：

    **标题**：{title}

    **详细描述**：{detail}

    ### 需要提取的要点
    1. 研究目的 / 作用机制与 {gene} 关联
    2. 干预措施（药物、剂量、联合方案）
    3. 适应症 & 纳入人群
    4. 研究设计 (Phase, Randomization, Masking, Arms)
    5. 当前状态与关键时间节点
    """)
