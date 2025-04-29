# llm_provider.py
import dotenv
import os
from pydantic import BaseModel
from typing import Literal
from langchain_core.language_models.chat_models import BaseChatModel # Import để type hint
dotenv.load_dotenv()

# --- Model Getters ---
def get_llm(model: Literal["qwen3:14b", "qwen3:30b", "qwq", "Qwen/QwQ-32B"] = "qwen3:14b", temperature: float = 0, **kwargs) -> BaseChatModel:
    """"""
    try:
        if model == "Qwen/QwQ-32B":
            from langchain_together import ChatTogether
            api = os.environ["TOGETHER_API_KEY"]
            if not api:
                raise "Together API key not found" 
            llm = ChatTogether(
                model=model,
                temperature=temperature,
                **kwargs
            )
        
        else:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model = model,
                temperature=temperature,
                repeat_penalty=1, top_p=0.95, top_k=20,
                **kwargs
            )
        
        return llm
    except Exception as e:
        print(f"Lỗi khi khởi tạo: {e}")
        raise e # Re-raise lỗi để báo hiệu vấn đề

def get_llm_structured(
    schema: type[BaseModel],
    model: Literal["qwen3:14b", "qwen3:30b", "qwq", "Qwen/QwQ-32B"] = "qwen3:14b", 
    temperature: float = 0,
    **kwargs
) -> BaseChatModel:

    llm = get_llm(model=model, temperature=temperature, **kwargs)
    structured_llm = llm.with_structured_output(schema=schema)
    return structured_llm
