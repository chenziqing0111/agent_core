# agent_core/state_machine/graph_definition.py

from langgraph.graph import StateGraph, END

def start_node(state):
    """开始节点"""
    state["next_node"] = "parse_user_input"
    return state

def parse_user_input(state):
    """解析用户输入（主控 agent 已完成）"""
    return state

def build_task_graph(state):
    """构建任务图（主控 agent 已完成）"""
    return state

def task_dispatch(state):
    """任务调度，检查任务列表并决定下一个任务"""
    if not state.get("tasks_to_run"):
        state["next_node"] = "combine_results"
    else:
        state["next_node"] = state["tasks_to_run"].pop(0)  # 弹出下一个任务
    return state

def combine_results(state):
    """合并各个代理结果并生成最终报告"""
    full_report = f"任务: {state.get('parsed_task', {}).get('task_name', '')}\n"
    full_report += f"文献摘要：{state.get('literature_result', '')}\n"
    full_report += f"网络信息：{state.get('web_result', '')}\n"
    full_report += f"竞品信息：{state.get('commercial_result', '')}\n"
    full_report += f"专利分析：{state.get('patent_result', '')}"
    state["final_report"] = full_report
    return state

def build_graph_with_nodes(node_mapping: dict):
    """构建任务图并返回"""
    graph = StateGraph()

    # 添加节点
    for node_name, node_func in node_mapping.items():
        graph.add_node(node_name, node_func)

    graph.set_entry_point("start_node")
    graph.add_edge("start_node", "parse_user_input")
    graph.add_edge("parse_user_input", "build_task_graph")
    graph.add_edge("build_task_graph", "task_dispatch")
    graph.add_edge("literature_agent", "task_dispatch")
    graph.add_edge("web_agent", "task_dispatch")
    graph.add_edge("commercial_agent", "task_dispatch")
    graph.add_edge("patent_agent", "task_dispatch")
    graph.add_edge("combine_results", END)

    return graph.compile()
