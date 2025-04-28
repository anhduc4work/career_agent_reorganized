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


class AgentState(MessagesState):
    cv: str
    jd: str
    sender: str
    new_cv: str
    chat_history_summary: str = ""
    last_index: int = 0

    
    

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

        WINDOWSIZE = 6
        MINNEWMESSAGESALLOW = 4

        if len(not_sum_messages) >= MINNEWMESSAGESALLOW + WINDOWSIZE:
            yield [Command(goto="extract_user_info"), Command(goto="filter_&_summarize_messages")]

    def _extract_user_info(self, state, config: RunnableConfig, store):
        """Use structured LLM output to extract user memory from new messages."""
        print("---EXTRACT USER INFO---")

        # Define the schema for user memory
        extractor = get_llm_structured(Memory)

        # Retrieve user ID and namespace
        user_id = config["configurable"].get("user_id", "")
        namespace = ("user_info", user_id)

        # Get messages not yet processed
        messages_to_process = state["messages"][state.get("last_index", 0):]
        if not messages_to_process:
            return
        conversation_str = "\n".join(f"{msg.type}: {msg.content}" for msg in messages_to_process)

        # current memory
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

        Here is a new conversation that may include updates or new details:

        {conversation_str}

        Please extract or update the user's memory profile.
        Only fill in what you are confident about.
        Don't erase or fabricate information unless new data clearly overrides the old.
        """.strip()
        # Call LLM extractor
        extracted = extractor.invoke([
                SystemMessage(extract_instruction),
                HumanMessage("Go on")
            ]
        )
        print("extracted: ",extracted)
        # Save extracted memory to store
        try: 
            store.put(namespace, "info" , extracted.model_dump_json())
        except Exception as error:
            print(error)
            pass
        
        
    def _filter_and_summarize_messages(self, state, config: RunnableConfig, store):
        print("---FILTER & SUMMARIZE---")
        all_messages = state["messages"]
        last_index = state.get("last_index") or 0
        not_sum_messages = all_messages[last_index:]

        WINDOWSIZE = 6
        messages_to_sum = not_sum_messages[:-(WINDOWSIZE)]
        conversation_str = "\n".join(f"{msg.type}: {msg.content}" for msg in messages_to_sum)
        new_last_index = last_index + len(messages_to_sum)

        current_summary = state.get("chat_history_summary", "")
        model = get_llm()

        updated_summary = model.invoke([
            SystemMessage(self.memo_instruction.format(
                current_memo=current_summary,
                conversation=conversation_str
            )),
            HumanMessage("Do it")
        ])

        user_id = config["configurable"].get("user_id", "")
        namespace = ("chat_history", user_id)

        if user_id:
            for m in messages_to_sum:
                store.put(namespace, m.id, {"data": f"{m.type}: {m.content}"})

        return Command(update={
            "chat_history_summary": updated_summary.content,
            "last_index": new_last_index
        })

    def _main_agent(self, state, config: RunnableConfig, store):
        print("---CALL AGENT---")
        last_index = state.get("last_index", 0)
        messages = state["messages"][last_index:]
        user_id = config["configurable"].get("user_id", "")
        config["recursion_limit"] = 2

        if not messages:
            return Command(update={"messages": [AIMessage("No message provided.")]}, goto=END)

        if isinstance(messages[-1], ToolMessage):
            try:
                error = json.loads(messages[-1].content).get("error", "")
                if error:
                    return Command(update={"messages": [AIMessage(error)]}, goto=END)
            except Exception:
                pass

        thread_memo = state.get("chat_history_summary", "")
        if thread_memo:
            print("memo:", thread_memo)

        # model = get_llm(model = "Qwen/QwQ-32B")
        model = get_llm()
        model = model.bind_tools(self.tools)

        
        namespace = ("user_info", user_id)
        try:
            user_info = store.get(namespace, "info")
            if user_info:
                user_info = user_info.value
                print("user_info:", user_info)
        except Exception:
            print("Fail to get user info")
            user_info = ""

        response = model.invoke([
            SystemMessage(self.agent_instruction.format(
                tool_names="\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                cv=state.get("cv", ""),
                jd=state.get("jd", ""),
                user_info=user_info,
                thread_memory=thread_memo
            ))
        ] + messages)
        
        print("cv: ", state.get("cv", "-------------")[:10])
        print(response)
        

        return Command(update={"messages": [response], "sender": "agent"})
    
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
        self.memo_instruction = """
You are summarization expert. Combine the current summary and the given conversation into only brief summary.
Remember to keep new summary short, brief in about 10-40 words, as short as possible.
Here is the current summarization (it can be empty):
{current_memo}
Here is the conversation to sum up:
{conversation}
""".strip()

        self.agent_instruction = """
You are a helpful AI assistant, collaborating with other assistants.
Use the provided tools to progress towards answering the question.
If you are unable to fully answer, that's OK, another assistant with different tools 
will help where you left off. Execute what you can to make progress.
If you or any of the other assistants have the final answer or deliverable,
prefix your response with FINAL ANSWER so the team knows to stop.
You should return data in table markdown for easily interpretation (for task relating comparation)
You have access to the following tools: 
{tool_names}
Here is the content of curriculum vitate of user (this is empty when user haven't uploaded it yet):
{cv}
Here is the content of job description that user mannually upload (this can be empty is they haven't upload):
{jd}
Here is your summary of recent chat with user: {thread_memory}
Here is your memory (it may be empty): {user_info}
If you have memory for this user, use it to personalize your responses.
""".strip()

        workflow = StateGraph(AgentState)
        workflow.add_node("router", self._router)
        workflow.add_node("extract_user_info", self._extract_user_info)
        workflow.add_node("filter_&_summarize_messages", self._filter_and_summarize_messages)
        workflow.add_node("agent", self._main_agent)
        workflow.add_node("tools", self.tool_node)

        workflow.set_entry_point("router")
        workflow.add_edge("router", "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

        self.graph = workflow.compile(checkpointer=self.checkpointer, store=self.store)

    def get_graph(self):
        return self.graph