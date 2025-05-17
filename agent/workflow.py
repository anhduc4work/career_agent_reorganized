from agent.sub_agents.coordinator import coordinator_node
from agent.sub_agents.job_searcher_agent import job_searcher, router
from agent.sub_agents.cv_agent import CVExpert
from agent.sub_agents.jd_agent import JDExpert
from agent.sub_agents.schema import AgentState
from langgraph.graph import START, END, StateGraph




# workflow = StateGraph(AgentState)
# workflow.add_node("coordinator", coordinator_node)
# workflow.add_node("job_searcher_agent", job_seacher)
# # workflow.add_node("router", router)
# workflow.add_node("cv_agent", CVExpert)
# workflow.add_node("jd_agent", JDExpert)


# workflow.set_entry_point("coordinator")

# from langgraph.checkpoint.memory import MemorySaver
# memory = MemorySaver()
# career_agent = workflow.compile(checkpointer=memory)








import json
import uuid
from pydantic import BaseModel, Field
from psycopg import connect
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.types import Command
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from .llm_provider import get_llm_structured, get_llm
from operator import add



from typing import Annotated, Optional, Union

    
class Memory(BaseModel):
    name: str | None = Field(
        default=None,
        description="Full name of the user, if provided in the conversation."
    )
    email: str | None = Field(
        default=None,
        description="Email address of the user, if mentioned."
    )
    phone: str | None = Field(
        default=None,
        description="Phone number of the user."
    )
    location: str | None = Field(
        default=None,
        description="Location or city the user is based in or prefers to work in."
    )
    career_goal: str | None = Field(
        default=None,
        description="Career aspirations or long-term goals shared by the user."
    )
    preferred_roles: list[str] = Field(
        default_factory=list,
        description="List of job titles or roles the user is interested in."
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Technical or soft skills mentioned by the user."
    )
    experience_summary: str | None = Field(
        default=None,
        description="Short summary of user's past experience if they mention it."
    )
    achievements: list[str] = Field(
        default_factory=list,
        description="Any notable achievements or projects the user talks about."
    )
    education_background: str | None = Field(
        default=None,
        description="User's education background, degrees or schools mentioned."
    )
    availability: str | None = Field(
        default=None,
        description="Availability to start working (e.g. immediately, in 2 weeks)."
    )
    preferences: str | None = Field(
        default=None,
        description="Any other preferences mentioned by the user, e.g., remote work, specific industries."
    )
    
    


class CareerAgent:
    def __init__(self, tools, pg_uri: str):
        self.tools = tools
        self.pg_uri = pg_uri
        self.checkpointer = None
        self.store = None
        self.graph = None
        self.tool_node = ToolNode(
            self.tools,
            name="tools",
            handle_tool_errors=self._handle_error
        )

    def _handle_error(self, error="") -> str:
        return json.dumps({'error': str(error)}) if error else json.dumps({'error': "unknown error"})

    def _router(self, state):
        # Decide if we need to extract info and summarize
        print("---ROUTER---")
        all_messages = state["messages"]
        last_index = state.get("last_index") or 0
        not_sum_messages = all_messages[last_index:]
        print("router msg", len(all_messages), "not sum", len(not_sum_messages), "indez", last_index)

        print("all messages: ", [m.content for m in all_messages])
        WINDOWSIZE = 6
        MINNEWMESSAGESALLOW = 4
        
        # yield Command(goto="agent")
        
        if len(not_sum_messages) >= MINNEWMESSAGESALLOW + WINDOWSIZE:
            yield [Command(goto="extract_user_info"), Command(goto="filter_&_summarize_messages")]

    def _extract_user_info(self, state, config: RunnableConfig, store):
        """Use structured LLM output to extract user memory from new messages."""
        print("---EXTRACT USER INFO---")

        extractor = get_llm_structured(Memory)
        user_id = config["configurable"].get("user_id", "")
        namespace = ("user_info", user_id)

        # Get messages not yet processed
        messages_to_process = state["messages"][state.get("last_index", 0):]
        if not messages_to_process:
            print("no messages to extract")
            return 
        conversation_str = "\n".join(f"{msg.type}: {msg.content}" for msg in messages_to_process if msg.type == 'user')

        try:
            current_memory = store.get(namespace, "info")
            if current_memory:
                current_memory = current_memory.value
            else:
                current_memory = ""
        except Exception:
            print("Fail to get user info")
            current_memory = ""
        
        extract_instruction = f"""
        You are a memory extractor that helps build user profile data.
        Here is the current memory (if any):

        {current_memory}

        Please extract or update the user's memory profile.
        Only fill in what you are confident about.
        Don't erase or fabricate information unless new data clearly overrides the old.
        """.strip()

        extracted = extractor.invoke([
                SystemMessage(extract_instruction),
                HumanMessage(f"Here is a new conversation that may include updates or new details:\n {conversation_str} /no_think")
            ]
        )
        print("extracted: ", extracted)
        # Save extracted memory to store
        try: 
            store.put(namespace, "info" , extracted.model_dump_json())
        except Exception as error:
            print("error:", error)
            pass
        
        
    def _filter_and_summarize_messages(self, state, config: RunnableConfig, store):
        print("---FILTER & SUMMARIZE---")
        all_messages = state["messages"]
        last_index = state.get("last_index") or 0
        not_sum_messages = all_messages[last_index:]

        WINDOWSIZE = 6
        messages_to_sum = not_sum_messages[:-(WINDOWSIZE)]
        conversation_str = "\n".join(f"{msg.type}: {msg.content[:100]}" for msg in messages_to_sum if msg.type != 'tool')
        new_last_index = last_index + len(messages_to_sum)

        current_summary = state.get("chat_history_summary", "")
        
        class SummarizeOutput(BaseModel):
            updated_summary: str = Field(..., description="Tóm tắt đã được cập nhật sau khi bao gồm các messages mới")
        model = get_llm_structured(SummarizeOutput)

        response = model.invoke([
            SystemMessage(self.memo_instruction.format(
                current_memo=current_summary,
            )),
            HumanMessage(f"Here is the new conversation to sum up: {conversation_str} /no_think")
        ])

        user_id = config["configurable"].get("user_id", "")
        namespace = ("chat_history", user_id)

        # if user_id:
        #     for m in messages_to_sum:
        #         store.put(namespace, m.id, {"data": f"{m.type}: {m.content}"})

        return Command(update={
            "chat_history_summary": response.updated_summary,
            "last_index": new_last_index
        })

    def _main_agent(self, state, config: RunnableConfig, store):
        print("---CALL AGENT---")
        
        last_index = state.get("last_index", 0)
        messages = state["messages"][last_index:]
        user_id = config["configurable"].get("user_id", "")
        thread_memory = state.get("chat_history_summary", "")
        cv = state.get("cv", "")
        jd = state.get("jd", "")
        print("len msgs: ", len(messages))
        config["recursion_limit"] = 2
        
        
        mode = state.get("sender", "think")
        if isinstance(messages[-1], ToolMessage):
            if mode == "no_think":
                messages.append(HumanMessage("/no_think"))
            else:
                pass
        elif isinstance(messages[-1], HumanMessage):
            if messages[-1].content.endswith('/no_think'):
                mode = 'no_think'
            else:
                mode = "think"
        else:
            pass
        
        
        model = get_llm(mode=mode)
        model = model.bind_tools(self.tools) # cause non streaming


        namespace = ("user_info", user_id)
        try:
            user_info = store.get(namespace, "info")
            if user_info:
                user_info = user_info.value
                print("user_info:", user_info)
            else:
                print("no user info")
        except Exception:
            print("Fail to get user info")
            user_info = ""

        system_prompt = self.agent_instruction.format(
                cv=cv,
                user_info=user_info,
                thread_memory=thread_memory
            )
        
        response = model.invoke([SystemMessage(system_prompt)]+ messages)
        
        return Command(update = {"messages": [response], "sender": mode})
    
    def setup_memory_and_store(self):
        conn = connect(self.pg_uri, autocommit=True)
        self.store = PostgresStore(conn, index={
            "embed": OllamaEmbeddings(model='nomic-embed-text'),
            "dims": 768,
        })
        self.store.setup()

        self.checkpointer = PostgresSaver(conn)
        self.checkpointer.setup()

    

    def build(self):
        
        
        workflow = StateGraph(AgentState)
        workflow.add_node("coordinator", coordinator_node)
        # workflow.add_node("router", router)
        workflow.add_node("cv_agent", CVExpert)
        workflow.add_node("job_searcher_agent", job_searcher)
        workflow.add_node("jd_agent", JDExpert)


        workflow.set_entry_point("coordinator")
        

        self.graph = workflow.compile(checkpointer=self.checkpointer, store=self.store)

    def get_graph(self):
        return self.graph