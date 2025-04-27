import os
from typing import Optional, Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
from enum import Enum
from typing import Optional, Annotated

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

# ---------------------------- TOOLS ----------------------------

@tool
def job_search_by_query(job: str, k: int = 3,
               job_type: Optional[JobType] = None,
               position: Optional[Position] = None) -> list[str]:
    """
    Search for jobs based on a query with optional filters.
    """
    print("--tool: job_search_by_query--")
    # print(job, k, job_type, position)

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

    return retriever.invoke(job)


@tool
def job_search_by_cv(cv: Annotated[str, InjectedState("cv")],
                     k: int = 3,
                     job_type: Optional[JobType] = None,
                     position: Optional[Position] = None) -> list[str]:
    """
    Search for jobs based on a provided CV with optional filters.
    """
    print("--tool: job_search_by_cv--")
    # print(k, job_type, position)

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

    return retriever.invoke(cv)