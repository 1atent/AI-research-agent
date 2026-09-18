import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def create_client() -> tuple[OpenAI, str]:
    """Create the shared OpenAI-compatible client and return its model name."""
    api_key = os.getenv("API_KEY")
    model = os.getenv("MODEL")
    if not api_key:
        raise RuntimeError("缺少环境变量 API_KEY。")
    if not model:
        raise RuntimeError("缺少环境变量 MODEL。")

    base_url = os.getenv("BASE_URL") or None
    return OpenAI(api_key=api_key, base_url=base_url), model
