# agent_core/agents/control_agent.py

from typing import Dict, List
from agent_core.clients.llm_client import LLMClient  # 导入 LLMClient
from agent_core.prompts.control_agent_prompts import get_task_description_prompt  # 导入新的 prompt 模块
from agent_core.state_machine.graph_runner import run_task_graph  # 导入新的任务图执行逻辑

class ControlAgent:
    def __init__(self, api_key: str, base_url: str):
        # 初始化状态
        self.state = {
            "user_input": "",
            "parsed_task": [],
            "tasks_to_run": [],
            "literature_result": "",
            "web_result": "",
            "commercial_result": "",
            "final_report": "",
            "dialog_history": [],
        }
        # 实例化 LLMClient
        print('yes')
        self.llm_client = LLMClient(api_key=api_key, base_url=base_url)

    def add_to_dialog_history(self, user_input: str, agent_response: str) -> None:
        """记录用户和系统的对话"""
        self.state["dialog_history"].append(f"用户: {user_input}")
        self.state["dialog_history"].append(f"LLM: {agent_response}")

    def ask_user_for_input(self) -> None:
        """与用户交互，获取需求，并解析任务"""
        print("我可以帮助你生成一个靶点研究报告。你希望报告关注哪些方面？")
        user_input = input("请输入你的需求：")

        # 调用 LLM 解析用户输入
        agent_response = self.ask_llm_with_history(user_input)

        # 保存对话历史
        self.add_to_dialog_history(user_input, agent_response)

        # 解析用户需求，动态生成任务
        self.state["parsed_task"] = self.parse_llm_response(agent_response)

        # 将任务添加到任务队列
        self.state["tasks_to_run"] = [task["task_name"] for task in self.state["parsed_task"]]

    def ask_llm_with_history(self, user_input: str) -> str:
        """使用 LLM 解析用户输入，并保留对话上下文"""
        history = "\n".join(self.state["dialog_history"])

        # 使用新的prompt模块
        prompt = get_task_description_prompt(history, user_input)

        # 调用 LLMClient 来获取响应
        return self.llm_client.chat_completion(messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ])

    def parse_llm_response(self, response: str) -> List[Dict]:
        """解析 LLM 返回的任务列表"""
        # 假设 LLM 返回的是类似这样的任务列表：
        # [{'task_name': '疾病', 'description': '与疾病相关的靶点研究'}, {'task_name': '靶点', 'description': '靶点研究'}]
        try:
            tasks = eval(response)  # 转换字符串为字典列表
            return tasks
        except:
            return []

    def build_initial_state(self) -> Dict:
        """返回初始状态"""
        return self.state

    def run(self) -> Dict:
        """启动控制流程，向用户提问并启动任务图"""
        self.ask_user_for_input()  # 与用户交互获取输入

        # 执行任务图（在 state_machine 中定义）
        result = run_task_graph(self.state)
        
        return result
