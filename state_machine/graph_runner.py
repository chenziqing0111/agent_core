# agent_core/state_machine/graph_runner.py

# 导入在 graph_definition.py 中定义的节点函数
from agent_core.state_machine.graph_definition import (
    start_node, 
    parse_user_input, 
    build_task_graph, 
    task_dispatch, 
    combine_results
)

from agent_core.agents.literature_agent import literature_agent
from agent_core.agents.web_agent import web_agent
from agent_core.agents.commercial_agent import commercial_agent
from agent_core.agents.patent_agent_wrapper import patent_agent
from agent_core.state_machine.graph_definition import build_graph_with_nodes

def run_task_graph(initial_state):
    """运行任务图，传入初始状态并执行任务流程"""
    # 定义所有节点
    node_mapping = {
        "start_node": start_node,
        "parse_user_input": parse_user_input,
        "build_task_graph": build_task_graph,
        "task_dispatch": task_dispatch,
        "literature_agent": literature_agent,
        "web_agent": web_agent,
        "commercial_agent": commercial_agent,
        "patent_agent": patent_agent,
        "combine_results": combine_results
    }

    # 构建任务图
    graph = build_graph_with_nodes(node_mapping)

    # 执行任务流
    result = graph.invoke(initial_state)

    return result
