# llm_provider.py
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel # Import để type hint
dotenv.load_dotenv()

# --- Gemini Model Getters ---

def get_llm(model: str = "gemini-1.5-flash", temperature: float = 0, **kwargs) -> BaseChatModel:
    """"""
    try:
        if model == "gemini-1.5-flash":
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-8b",
                temperature=temperature,
                **kwargs
            )
        else:
            llm = ChatTogether(
                model="Qwen/QwQ-32B",
                temperature=temperature,
                **kwargs
            )
        return llm
    except Exception as e:
        print(f"Lỗi khi khởi tạo ChatGoogleGenerativeAI: {e}")
        # Có thể raise lỗi hoặc trả về None tùy theo cách xử lý mong muốn
        raise e # Re-raise lỗi để báo hiệu vấn đề

def get_llm_structured(
    schema: type[BaseModel],
    model: str = "gemini-1.5-flash", # Model hỗ trợ tốt structured output/function calling
    temperature: float = 0,
    **kwargs
) -> BaseChatModel:

    llm = get_llm(model=model, temperature=temperature, **kwargs)
    structured_llm = llm.with_structured_output(schema=schema)
    return structured_llm
