from pydantic import BaseModel, Field
from typing import Annotated, Optional, Union, Literal
from langgraph.graph import MessagesState, StateGraph, END
from agent.llm_provider import get_llm_structured, get_llm
from agent.tools.retrieve_pg_tools import vector_store
from langgraph.constants import Send
from langgraph.types import Command
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage
from operator import add
from pydantic import BaseModel, Field
from typing import Literal
from typing import Optional, Literal, List, Union, Dict, get_args
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool
from langgraph.types import Command
from agent.llm_provider import get_llm_structured, get_llm
from agent.tools.retrieve_pg_tools import vector_store
from langgraph.constants import Send
from typing import Optional, Literal, List, Union, Dict, get_args
import os 
import json







class CVExpertOutput(BaseModel):
    # Định tuyến tác vụ đến agent xử lý hình thức hoặc nội dung CV
    next_step: Literal['cv_format', 'cv_content'] = Field(
        description=(
            "Determines the next specialized agent to handle the request:\n"
            "- 'cv_format': handles formatting and presentation of the CV (by default if user not mention anything)\n"
            "- 'cv_content': handles content relevance and alignment with job description"
            """# Rules

            ## 1. Next Step: CV Type
            - **"cv_format"**:
                - The user talks about appearance, layout, styling, visual structure, or general professionalism.
                - Or: The user does *not mention* any job, JD, role, position, or matching.
                - Or: The user just say review/rewrite my cv.

            - **"cv_content"**:
                - The user mentions relevance, matching, alignment with roles, job descriptions, or skills."""
        )
    )

    # Loại yêu cầu: đánh giá hay chỉnh sửa CV
    action_type: Literal['review', 'rewrite'] = Field(..., description="""Indicates whether the request is for 'review' (assessment only) or 'rewrite' (content update/improvement) Action Type
            - **"review"**:
                - The user wants an evaluation, feedback, or opinion only.
                - Clues: "check", "review", "is it good enough", "does it match", etc.

            - **"rewrite"**:
                - The user wants to rewrite, improve, optimize, or tailor the CV.
                - Clues: "rewrite", "improve", "make better", "customize", etc.""")

    # Chỉ mục (index) của job để dùng làm đối chiếu với CV
    jd_index: int = Field(..., description="""
                          Index or ID of the Job Description to compare against when analyzing CV content. If none: set to 4939 ## 3. JD Index
            - If the request mentions a specific job, role, or reference number, extract the correct JD index (e.g. 7383).
            - If no JD is mentioned or implied, return 0.""")

CV_SYSTEM_PROMPT = """
You are an expert assistant that routes CV-related requests to the correct processing step.

Your job is to determine:
1. Whether it concerns **format** (layout, style) or **content** (matching a job description).
2. Whether the user wants a **review** (feedback only) or a **rewrite** (improvement).
3. Which job description to use for comparison, if applicable.

# Example
User: "Review my cv"
→ { "next_step": "cv_format", "action_type": "review", "jd_index": "null" }

User: "adjust/align/improve my cv for the job role (JD #4920)"
→ { "next_step": "cv_content", "action_type": "rewrite", "jd_index": 4920 }

# Notes
- Always return all 3 fields.
"""
def cv_expert(state) -> Command:
    print("--cv--")
    print("----",type(state), state)
    jd = state.get('jd', '')
    
    if not state.get('cv', ''):
        return Command(
            goto = state['sender'],
            update={"messages": [AIMessage('CV is not uploaded yet')],'sender': 'cv_expert'},
        )
            
    llm = get_llm_structured(CVExpertOutput)
    response = llm.invoke([SystemMessage(CV_SYSTEM_PROMPT)] + state["messages"])
    print('response: ', response)
    
    if response.next_step == 'cv_content':
        
        feedback =     'content_reviewer_insights'
    else:
        feedback =     'format_reviewer_insights'
        
    print('----', feedback, '----')
    if response.action_type == 'rewrite' and state.get(feedback, ''):   
        print('da danh gia truoc roi, chuyen qua viet luon') 
        return Send('cv_writer', {'sender': feedback,
                                    'goto': response.action_type,
                                    # 'jd': jd,
                                    'cv': state['cv'],
                                    feedback: state.get(feedback, '')
                                    })
    elif response.action_type == 'review' and state.get(feedback, ''):   
        print('da danh gia truoc roi, chuyen qua viet luon') 
        return Send('cv_writer', {'sender': feedback,
                                    'goto': response.action_type,
                                    # 'jd': jd,
                                    'cv': state['cv'],
                                    feedback: state.get(feedback, '')
                                    })
    else:
                
        if response.next_step == 'cv_content':
            
            if not jd:
                try:
                    jd = vector_store.get_by_ids([str(response.jd_index)])[0].page_content
                    print(f'got {jd}')
                except Exception as e:
                    jd = vector_store.get_by_ids(['4942'])[0].page_content
                    print(e)
            
            return Send('jd_extractor', {'sender': 'cv_expert','goto': response.action_type,
                                    'jd': jd, 'cv': state['cv']})

        return Send(response.next_step, {'sender': 'cv_expert','goto': response.action_type, 'cv': state['cv']})
                

from .schema import AgentState
def format_reviewer(state: AgentState) -> Command:
    
    FORMAT_REVIEW_SYSTEM_PROMPT = """
You are an HR expert whose task is to review and evaluate a candidate’s CV based on the following detailed criteria. Carefully read and analyze the CV, then provide clear, actionable feedback. Highlight specific errors and offer direct suggestions for improvement. Avoid vague or generic comments. Follow these evaluation points:
	1.	Relevance to the job:

	•	Does the CV show a clear connection to the target job?
	•	Are relevant skills, education, and keywords included?

	2.	Timeline structure:

	•	Are all time periods for education and work fully listed?
	•	Is the timeline sorted in reverse chronological order?
	•	Are there unexplained time gaps?

	3.	Personal information:

	•	Is the candidate’s name clearly and prominently displayed?
	•	Is the email professional (ideally containing the name)?
	•	Is the phone number formatted for easy reading (e.g., 0123 456 789)?
	•	Are social media links appropriate and professional (prefer LinkedIn over Facebook)?
	•	Is the address overly detailed? (Only district/city is needed)
	•	Are there unnecessary details like ID number, gender, age, or marital status?
	•	Is the personal info section concise (less than 4 lines)?

	4.	Career objective (optional):

	•	Is the career objective clear and concise (maximum 4 lines)?
	•	Is it personalized, specific, and not generic?

	5.	Education:

	•	Does it include institution name, major, and start/end dates?
	•	Is the order correct (most recent first)?
	•	Are relevant courses or certifications beyond formal degrees listed?

	6.	Skills:

	•	Is there a skills section?
	•	Are the skills relevant to the job?
	•	Are soft skills and technical skills clearly differentiated?
	•	Are rating bars used without clear criteria? If yes, remove or replace with measurable standards (e.g., IELTS 6.5).
	•	Are there 4–8 skills listed? Too few or too many may be ineffective.
	•	Are action verbs used to describe the skills? (e.g., “Manage a team” instead of “Leadership”)

	7.	Work experience & activities:

	•	Is the timeline sorted from most recent to oldest?
	•	Are there unexplained gaps?
	•	Does each job include a short description of the company (size, product, industry)?
	•	Are there measurable results (numbers, KPIs, achievements)?
	•	Does each role have at least 3 bullet points describing tasks and outcomes?

	8.	References:

	•	Are references only included when requested?
	•	If listed, do they contain full name, position, company, email, and phone number?

	9.	Hobbies:

	•	Do the listed hobbies help the candidate stand out?
	•	Are they specific (e.g., reading psychology books, listening to classical music)?

	10.	Overall layout and formatting:

	•	Is the structure logical? (Personal Info > Skills > Experience > Education)
	•	Is the length appropriate (between 40–80 lines)?
	•	Are personal pronouns like “I”, “My”, or “Me” avoided?

After reviewing, return your feedback in the form of a checklist or structured list, pointing out both the problems and suggested improvements.
"""
    class Feedback(BaseModel):
        issue: str = Field(description="The issue identified by the human reviewer")
        solution: str = Field(description="The solution to the issue identified by the human reviewer")
        criteria: str = Field(description="The criteria used to evaluate the CV")
            
    class Feedbacks(BaseModel):
        feedbacks: list[Feedback] = Field(description="The feedbacks from the human reviewers")
    
    print('--review format--')
    if state['goto'] == 'review':
        # tra loi truc tiep
        llm = get_llm()
    else:
        llm = get_llm_structured(Feedbacks)
        
    response = llm.invoke([SystemMessage(FORMAT_REVIEW_SYSTEM_PROMPT), HumanMessage(f'Start review {state["cv"]} /no_think')])
    
    print('--done review format--')
    
    
    if state['goto'] == 'review':
        # end process
        return Command(
            goto = 'coordinator',
            graph = Command.PARENT,
            update={"message_from_sender": response,'sender': 'format_reviewer',
                    "format_reviewer_insights":response.content},
        )
    else:
        return Command(
            goto = 'cv_writer',
            update={'sender': 'format_reviewer', "format_reviewer_insights":response},
        )
        

def jd_extractor(state):
    EXTRACTOR_INSTRUCTION = """
# Role and Objective  
You are a Job Description (JD) analysis expert. Your mission is to deeply analyze the provided JD and extract key hiring signals that reveal the employer’s true expectations and priorities for the ideal candidate. You are not just listing what’s mentioned — you must infer deeper intent and structure your output clearly.

# Task  
Analyze the given job description and extract the most important requirements and signals. Your output must be paraphrased and organized into **five categories**:

## Output Format (Use exact structure):
1. **Technical Skills**  
List relevant tools, technologies, programming languages, frameworks, platforms, or systems explicitly or implicitly required.

2. **Soft Skills**  
List essential non-technical traits or behaviors (e.g. communication, teamwork, leadership, problem-solving).

3. **Experience Requirements**  
Summarize the years of experience, industry background, project types, or job roles needed.

4. **Education & Certifications**  
List any degrees, majors, educational levels, or certifications that are required or preferred.

5. **Hidden Insights (Recruiter Intent)**  
Reflect on the overall tone, emphasis, and structure of the JD to infer the *real* goals behind the role.  
- Why is this role open now?  
- What kind of personality or archetype fits best?  
- What challenges might this role be solving?

# Instructions  

- **Paraphrase only** — Do **not** copy-paste content from the JD.  
- **Think critically** — Focus on what matters most, not just what’s explicitly stated.  
- **Be concise** — Use clean, bullet-pointed lists that are easy to scan.  
- **Infer meaning** — Capture both overt and subtle signals from the JD.  

# Reasoning Steps  
1. Read the full JD carefully.  
2. Identify explicit requirements and implied expectations.  
3. Categorize the findings under the 5 output sections.  
4. Paraphrase everything professionally and clearly.  

Final Note

You are acting as a specialist whose job is to extract and interpret hiring intent behind job postings. Respond with only the structured output. Do not include any explanations, introductions, or summaries outside of the 5 required categories.
"""
    class ExtractorOutput(BaseModel):
        technical_skills: List[str] = Field(
            ..., 
            description="List of tools, technologies, programming languages, frameworks, platforms, or systems required or implied."
        )
        soft_skills: List[str] = Field(
            ..., 
            description="List of essential non-technical traits or behaviors expected for the role."
        )
        experience_requirements: List[str] = Field(
            ..., 
            description="Summarized expectations for years of experience, industries, project types, or job roles."
        )
        education_certifications: List[str] = Field(
            ..., 
            description="Degrees, majors, education levels, or certifications that are required or preferred."
        )
        hidden_insights: List[str] = Field(
            ..., 
            description="Inferred recruiter intent such as purpose of the role, ideal personality, timing, or underlying organizational needs."
        )
    
    print('--extract--')
    print(type(state), state.keys())
    
    
    structured_llm = get_llm_structured(ExtractorOutput)
    response = structured_llm.invoke(
        [SystemMessage(EXTRACTOR_INSTRUCTION), HumanMessage(f'Start extracting {state["jd"]}')]
    )
    print("--done extract--")
    # return Send('analyze_cv', {"extractor_insights": response, "curriculum_vitae": state["curriculum_vitae"]})
    return {"extractor_insights": response, 'goto': state["goto"]}
        
def cv_analyst(state):
    ANALYST_INSTRUCTION = """# Role and Objective
You are a career analysis expert. Your mission is to evaluate how well a candidate's CV aligns with the key hiring criteria extracted from a job description (JD).

You will be given:
- A list of **JD Insights** (divided into 5 categories).
- The candidate’s **CV** (in text form).

# Task
For each of the 5 JD insight categories listed below, assess the CV and provide structured feedback.

## Output Format (Use exact structure):

Return a list of 5 feedback items, each containing:

1. **name**  
One of the following exact values:
   - `technical_skills`
   - `soft_skills`
   - `experience_requirements`
   - `education_certifications`
   - `hidden_insights`

2. **score**  
Rate from 0 to 10 how well the CV fulfills the expectations for that category (higher is better).

3. **comment**  
Give a concise, professional evaluation. Explain how well the CV meets this requirement. Mention any gaps or strong points.

## Scoring Guide
- **9–10**: Fully satisfies the requirement; matches perfectly or exceeds expectations.
- **7–8**: Mostly meets expectations; minor gaps.
- **4–6**: Partially meets expectations; some important aspects missing.
- **1–3**: Barely relevant; major issues.
- **0**: Not addressed at all in the CV.

# Categories to Evaluate
You must review the CV according to the following categories:

1. **Technical Skills**  
Compare listed tools, technologies, platforms, or programming languages with what the JD expects.

2. **Soft Skills**  
Evaluate evidence of interpersonal traits like communication, teamwork, leadership, adaptability, etc.

3. **Experience Requirements**  
Check for years of experience, relevant industries, types of projects, and similar past roles.

4. **Education & Certifications**  
Look for relevant degrees, fields of study, or professional certifications.

5. **Hidden Insights (Recruiter Intent)**  
Assess whether the candidate seems to match the deeper intent behind the role (e.g. personality, purpose of the role, cultural fit).

# Instructions
- Be objective and professional.
- Justify the score with a short but insightful comment.
- Think critically. Don’t just match keywords — understand fit.
- Respond ONLY with the structured output, nothing more.

# Input Context

## JD Insights
{insights}
"""
    class Feedback(BaseModel):
        name: Literal[
            'technical_skills',
            'soft_skills',
            'experience_requirements',
            'education_certifications',
            'hidden_insights'
        ] = Field(..., description="The criteria being evaluated.")
        score: int = Field(..., description="Score from 0 to 10 reflecting how well the CV meets this requirement.")
        comment: str = Field(..., description="Brief comment analyzing how well the CV fulfills this requirement.")

    class AnalystOutput(BaseModel):
        feedbacks: List[Feedback] = Field(..., description="List of feedback items for each key requirement category.")
    print("--analyze--")
    print(type(state), state.keys())

    system_message = ANALYST_INSTRUCTION.format(
        insights = state["extractor_insights"]
    )

    structured_llm = get_llm_structured(AnalystOutput)
    response = structured_llm.invoke(
        [SystemMessage(system_message), HumanMessage(f'Start analyzing cv: {state["cv"]}')]
    )
    print("--done analyze--")

    # return Send('suggest_cv', {"insights": response, "curriculum_vitae": state["curriculum_vitae"]})
    return {"analyst_insights": response, 'goto': state["goto"]}
        


def content_reviewer(state: AgentState):
    
    SUGGESTOR_INSTRUCTION = """# Role and Objective
You are a professional CV improvement advisor. Your task is to review structured feedback about how well a CV matches a job description, and determine whether and how the CV can be improved.

Your job is to:
- Assess each requirement category based on the feedback
- Decide whether improvements are possible through rewriting or restructuring the CV
- Suggest how to improve it if possible
- Identify relevant keywords or phrases that could be added to enhance alignment with the JD

# Input
You will receive a list of 5 feedback entries. Each entry includes:
- `name`: the requirement category (e.g., "technical_skills")
- `score`: numeric score (0–10) reflecting how well the CV meets that requirement
- `comment`: a brief explanation of the score

# For each category, return the following:
1. `name`: the category name (keep unchanged)
2. `action_needed`: one of:
   - `"yes"` → Improvements are possible and recommended
   - `"no"` → Already strong, no change needed
   - `"cannot_be_improved"` → Gaps reflect real limitations that cannot be fixed through CV editing
3. `recommendation`: 
   - If `"yes"` → Suggest what to rephrase, highlight, or clarify in the CV
   - If `"no"` → Briefly note the strength
   - If `"cannot_be_improved"` → Explain why improvement is not possible through writing
4. `suggested_keywords`: 
   - A list of relevant terms (technologies, behaviors, job titles, etc.) that could be added to improve alignment
   - Leave empty (`[]`) if `action_needed` is `"no"` or `"cannot_be_improved"`

# Guidelines
- Be realistic: do not suggest faking credentials or experience.
- Think critically about what can actually be added, emphasized, or reworded in a CV.
- Use bullet points and keep output concise and professional.

# Input Context
**Feedback List:**
{insights}"""
    
    class ImprovementSuggestion(BaseModel):
        name: Literal[
            'technical_skills',
            'soft_skills',
            'experience_requirements',
            'education_certifications',
            'hidden_insights'
        ] = Field(..., description="The requirement category being evaluated.")
        
        action_needed: Literal[
            'yes',                 # CV can and should be improved for this aspect
            'no',                  # This aspect is already strong, no changes needed
            'cannot_be_improved'   # This gap reflects something that cannot realistically be improved through CV edits (e.g., lack of years of experience)
        ] = Field(..., description="Whether the CV can be improved in this area.")
        
        current_expression: str = Field(..., description="How the candidate has currently demonstrated or expressed this aspect in the CV. Paraphrase or quote relevant parts.")
        
        recommendation: str = Field(..., description="Suggestion on what to change, emphasize, or reword in the CV. If no action is needed or not possible, explain why.")
        
        suggested_keywords: List[str] = Field(..., description="List of relevant keywords (technologies, soft skills, etc.) to consider adding to improve alignment with the JD.")


    class SuggestorOutput(BaseModel):
        suggestions: List[ImprovementSuggestion] = Field(..., description="List of improvement suggestions for each requirement category.")


    print('--review content--')
    print(type(state), state.keys())
    

    system_message = SUGGESTOR_INSTRUCTION.format(
        insights = state["analyst_insights"]
    )
    if state['goto'] == 'review':
        llm = get_llm()
    
    else:
        llm = get_llm_structured(SuggestorOutput)
    response = llm.invoke(
        [SystemMessage(system_message), HumanMessage(f"Start reviewing cv: {state['cv']}")]
    )
    print('--done review content--')
    
    if state['goto'] == 'review':
        
        return Command(
            goto = 'coordinator',
            graph = Command.PARENT,
            
            update={"message_from_sender": AIMessage(response.content),'sender': 'content_reviewer',
                    "content_reviewer_insights": response},
        )
    else:
        return Command(
            goto = 'cv_writer',
            update={'sender': 'content_reviewer', "content_reviewer_insights": response},
        )
        
# workflow = StateGraph(AgentState)
# workflow.add_node("jd_extractor", jd_extractor)
# workflow.add_node("cv_analyst", cv_analyst)
# workflow.add_node("content_reviewer", content_reviewer)
# workflow.set_entry_point("jd_extractor")
# workflow.add_edge('jd_extractor',"cv_analyst")
# workflow.add_edge('cv_analyst',"content_reviewer")
# # workflow.set_finish_point("content_reviewer")
# ContentReviewer = workflow.compile()


def cv_writer(state):
    
    WRITER_INSTRUCTION = """
# Role and Objective  
You are a professional CV editing assistant. Your job is to improve the candidate’s CV to better align with a job description (JD), based on a structured improvement plan generated by a JD-CV analysis agent.

Your edits should make the CV more relevant, professional, and compelling — while staying truthful to the candidate’s background.

# Task  
Use the improvement suggestions provided to revise the CV. You may:
- Emphasize or reword existing experience
- Add missing but plausible details based on the candidate's background
- Inject relevant keywords or terminology from the job description
- Reorganize or clarify content for better readability

# Editing Rules  
- **Do not fabricate** experience that doesn’t exist (e.g., fake years of experience or degrees).
- You may reframe or emphasize relevant work to match JD expectations.
- Do not add certifications, education, or job titles the candidate does not have.
- Add relevant **keywords** where appropriate, based on suggestion.
- Be clear, concise, and use a professional tone.
- You can update or add bullet points, rephrase summaries, and enhance skill sections.

# Input  
You will receive the following:
1. The original CV text.
2. A list of improvement suggestions in structured format, each with:
   - The category (e.g. technical skills, soft skills)
   - Whether an action is needed
   - A recommendation
   - Suggested keywords

# Output  
Return the **revised CV text only** — do not include any commentary or explanation. The CV should reflect the edits and improvements based on the provided plan.

Improvement Plan:

{insights}

Final Note

Do not mention that this is an edited CV. Just return the improved CV text.
"""


    class WriterOutput(BaseModel):
        new_cv: str = Field(..., description="The fully rewritten and improved CV based on the provided suggestions. The output should be clean, professional, and align closely with the job description.")

        
    print('--writer--')
    print(type(state), state)
    curriculum_vitae = state["cv"]
    
    if state['sender'].startswith('content_reviewer'):
        insights = state.get('content_reviewer_insights', '')
    elif state['sender'].startswith('format_reviewer'):
        insights = state.get('format_reviewer_insights', '')
    else:
        insights = ''

    system_message = WRITER_INSTRUCTION.format(
        insights = insights
    )
    llm = get_llm()
    response = llm.invoke(
        [SystemMessage(system_message), HumanMessage(f'Start rewriting cv: {curriculum_vitae} /no_think')]
    )
    # extractor = get_llm_structured(WriterOutput)
    # final = extractor.invoke([HumanMessage(f"""The following message contains a rewritten CV. Please extract **only the full CV text** from it.
    # Message:
    # {response.content}""")])
    
    print('--done write--')
    
    return Command(
        goto = 'coordinator',
        graph = Command.PARENT,
        update={'sender': 'writer', "message_from_sender": response, 'new_cv': response.content},
    )


workflow = StateGraph(AgentState)
workflow.add_node("cv_expert", cv_expert)

workflow.add_node("cv_format", format_reviewer)

workflow.add_node("jd_extractor", jd_extractor)
workflow.add_node("cv_analyst", cv_analyst)
workflow.add_node("content_reviewer", content_reviewer)
workflow.add_edge('jd_extractor',"cv_analyst")
workflow.add_edge('cv_analyst',"content_reviewer")

workflow.add_node("cv_writer", cv_writer)

workflow.set_entry_point("cv_expert")
# workflow.set_finish_point("cv_writer")


from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
CVExpert = workflow.compile()