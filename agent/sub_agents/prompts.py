from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage
from typing import Optional, Literal, List, Union, Dict, get_args
from typing import Annotated, Optional, Union, Literal

from operator import add
from langgraph.graph import MessagesState, StateGraph, END
from pydantic import BaseModel, Field




class AgentState(MessagesState):
    sender: str
    cv: Optional[str] 
    jds: Annotated[list, add]
    sender: Optional[str]
    new_cv: Optional[str]
    chat_history_summary: str 
    last_index: int = 0
    jd: Optional[str] 
    extractor_insights: Optional[dict] 
    analyst_insights: Optional[dict] 
    suggestor_insights: Optional[dict] 
    goto: str
    content_reviewer_insights: str | dict 
    format_reviewer_insights: str | dict
    
    
class CoordinatorOutput(BaseModel):
    next_step: Literal['__end__', 'job_searcher_agent', 'cv_agent', 'jd_agent'] = Field(
        default='__end__',
        description=(
            "Next step in the graph flow. Determines which specialized agent will handle the request:\n"
            "- '__end__': End the interaction or continue small talk with CareerFlow.\n"
            "- 'job_searcher_agent': Hand off to Job Search Agent (e.g., search, filter, role exploration).\n"
            "- 'cv_agent': Hand off to CV Agent (e.g., CV review, job alignment, rewrite suggestions).\n"
            "- 'jd_agent': Hand off to JD Agent (e.g., scoring JD relevance, ranking jobs, synthesizing job trends)."
        )
    )
    message_to_user: str = Field(..., description="A friendly message to send to the user.")
    message_to_next_agent: str = Field(..., description=(
            "A summary or instruction to pass to the next agent. "
            "This should clearly explain the user's intent, request, or any necessary context. "
            "Leave empty if `next_step` is '__end__'."
        )
    )
    
    
class CVExpertOutput(BaseModel):
    # Định tuyến tác vụ đến agent xử lý hình thức hoặc nội dung CV
    next_step: Literal['cv_format', 'cv_content'] = Field(
        default='cv_format',
        description=(
            "Determines the next specialized agent to handle the request:\n"
            "- 'cv_format': handles formatting and presentation of the CV (by default if user not mention anything)\n"
            "- 'cv_content': handles content relevance and alignment with job description"
        )
    )

    # Loại yêu cầu: đánh giá hay chỉnh sửa CV
    action_type: Literal['review', 'rewrite'] = Field(..., description="Indicates whether the request is for 'review' (assessment only) or 'rewrite' (content update/improvement)")

    # Chỉ mục (index) của job để dùng làm đối chiếu với CV
    jd_index: str = Field(default='4943', description="Index or ID of the Job Description to compare against when analyzing CV content")


class Feedback(BaseModel):
    issue: str = Field(description="The issue identified by the human reviewer")
    solution: str = Field(description="The solution to the issue identified by the human reviewer")
    criteria: str = Field(description="The criteria used to evaluate the CV")
        
class Feedbacks(BaseModel):
    feedbacks: list[Feedback] = Field(description="The feedbacks from the human reviewers")
    
    
