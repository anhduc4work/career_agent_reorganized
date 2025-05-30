# llm_provider.py

import dotenv
from pydantic import BaseModel
from typing import Literal
from langchain_core.language_models.chat_models import BaseChatModel # Import để type hint
dotenv.load_dotenv()
import os

default_model = os.getenv("DEFAULT_MODEL", "qwen3:4b")
num_ctx = int(os.getenv("NUM_CTX", 4096))

def get_llm(
    model: Literal["qwen3:4b","qwen3:8b", "qwen3:14b", "qwen3:30b", "gpt-4o"] = default_model,
    mode: Literal["think", "non-think"] = "non-think",
    num_ctx = num_ctx,
) -> BaseChatModel:
    """
    Return a configured ChatOllama model.

    Args:
        model: Name of the model to use.
        mode: If 'think', use settings for deeper reasoning. Otherwise, use default non-think settings.

    Returns:
        BaseChatModel: An instance of ChatOllama with the desired configuration.
    """
    try:
        if model == 'gpt-4o':
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(model="gpt-4o")
        else: 
            from langchain_ollama import ChatOllama

            if mode == "think":
                temperature = 0.6
                top_p = 0.95
            else:  # non-think
                temperature = 0.7
                top_p = 0.8

            
            llm = ChatOllama(
                model=model,
                temperature=temperature, 
                top_p=top_p,
                top_k=20,
                repeat_penalty=1.1,
                num_ctx = num_ctx,
            )

        return llm

    
    except Exception as e:
        print(f"Lỗi khi khởi tạo: {e}, let use open ai")
        raise e


def get_llm_structured(
    schema: type[BaseModel],
    model: Literal["qwen3:4b","qwen3:8b", "qwen3:14b", "qwen3:30b", "qwq", "gpt-4o"] = default_model, 
    mode: Literal["think", "non-think"] = "non-think",
    **kwargs
) -> BaseChatModel:
    """
    Return a structured-output LLM with the given schema.

    Args:
        schema: Pydantic schema to enforce structure on output.
        model: Model name.
        mode: Think vs non-think mode.

    Returns:
        BaseChatModel: Structured-output LLM.
    """
    llm = get_llm(model=model, mode=mode, **kwargs)
    return llm.with_structured_output(schema=schema)