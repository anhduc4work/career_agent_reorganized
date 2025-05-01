import os
from typing import Optional, Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
from enum import Enum
from langgraph.types import Command
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
# ---------------------------- PROMPT ----------------------------

class JobType(str, Enum):
    parttime = "parttime"
    fulltime = "fulltime"
    negotiation = "negotiation"

class Position(str, Enum):
    Management = "Management/Leadership"
    Postdoc = "Postdoc Position"
    Teaching = "Teaching/Lecturer Position"
    Research = "Research Position"
    PhD = "PhD Scholarship"
    ProfessorTrack = "Assistant/Associate/Full Professor"
    Staff = "Staff/Technician/Engineer Position"
    Undergrad = "Undergraduate Scholarship"
    Other = "Other"
    Master = "Master Scholarship"
    Admin = "Administration/Managerment"
    Lecturer = "Lecturer Position"
    Professor = "Professor Position"
    Faculty = "Faculty Position"



# ---------------------------- VECTOR STORE SETUP ----------------------------

def get_vector_store():
    connection = os.getenv("PG_CONN", "postgresql+psycopg://postgres:postgres@localhost:5432/postgres")
    collection_name = os.getenv("PG_COLLECTION", "scholar2")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

vector_store = get_vector_store()
import json


def documents_to_json(documents, include_content: bool = True):
    result = []
    for doc in documents:
        item = {
            "id": str(doc.id) if hasattr(doc, "id") else None,
            "metadata": doc.metadata
        }
        if include_content:
            item["page_content"] = doc.page_content
        result.append(item)
    return result

# ---------------------------- TOOLS ----------------------------

@tool
def job_search_by_query(
    job: str, 
    tool_call_id: Annotated[str, InjectedToolCallId],
    include_content: bool = True,
    k: int = 3,
    job_type: Optional[JobType] = None,
    position: Optional[Position] = None,
) -> list[str]:
    """
    Search for job descriptions (JDs) relevant to a given job title or query.

    This tool finds related job descriptions based on keyword similarity and optional filters.

    Args:
        job (str): Job search query (e.g., job title or relevant keywords).
        include_content (bool): Whether to return full JD content. Set to False if content is not needed (e.g., for market analysis). Defaults to True.
        k (int, optional): Number of top matching jobs to return. Defaults to 3.
        job_type (Optional[JobType], optional): Filter by job type (e.g., fulltime, parttime, etc.).
        position (Optional[Position], optional): Filter by job level (e.g., junior, senior).
    
    Returns:
        list: List of job IDs matching the query.

    Tip:
        For market analysis, set k >= 5 to retrieve enough samples for meaningful insights.
    """
    
    print("--tool: job_search_by_query--")
    
    filters = {}
    if job_type:
        filters["workingtime"] = job_type.value
    if position:
        filters["position"] = position.value

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            'score_threshold': 0.5,
            "k": k,
            'filter': filters
        }
    )
    # Format result: 
    output = retriever.invoke(job)
    formated_response = documents_to_json(output, include_content)
    
    return Command(update={"messages": [ToolMessage(f"Here is the {len(output)} jobs founded: " + json.dumps(formated_response, indent=2, ensure_ascii=False), 
                                                    tool_call_id=tool_call_id)], "jds": formated_response})


@tool
def job_search_by_cv(
    cv: Annotated[str, InjectedState("cv")],
    tool_call_id: Annotated[str, InjectedToolCallId],
    k: int = 3,
    include_content: Optional[bool] = True,
    job_type: Optional[JobType] = None,
    position: Optional[Position] = None,
) -> list[str]:
    """
    Search for jobs based on the content of a CV, with optional filters.

    Args:
        k (int, optional): The number of top results to return. Defaults to 3.
        job_type (Optional[JobType], optional): Filter by job type (e.g., fulltime, parttime, negotiation).
        position (Optional[Position], optional): Filter by job position.
    """
    
    print("--tool: job_search_by_cv--")
    print(k, job_type, position)

    filters = {}
    if job_type:
        filters["workingtime"] = job_type.value
    if position:
        filters["position"] = position.value

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            'score_threshold': 0.5,
            "k": k,
            'filter': filters
        }
    )
    output = retriever.invoke(cv)
    formated_response = documents_to_json(output, include_content)
    
    return Command(update={"messages": [ToolMessage(f"Here is the {len(output)} jobs founded: " + json.dumps(formated_response, indent=2, ensure_ascii=False), 
                                                    tool_call_id=tool_call_id)], "jds": formated_response})
    
    