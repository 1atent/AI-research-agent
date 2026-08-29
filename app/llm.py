import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def ask_llm(question: str) -> str:
    response = client.chat.completions.create(
        model="你的模型名称",
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
    )

    return response.choices[0].message.content