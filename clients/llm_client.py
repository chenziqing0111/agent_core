# agent_core/clients/llm_client.py
from openai import OpenAI

client = OpenAI(api_key="sk-9b3ad78d6d51431c90091b575072e62f", base_url="https://api.deepseek.com")

def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "你是一个生物医学分析助手"},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content