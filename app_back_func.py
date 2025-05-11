from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage
import uuid
import time
import gradio as gr
# import dotenv
# dotenv.load_dotenv()
from agent.agent import CareerAgent
from agent.tools import all_tools as tools 

# Postgres connection URI
PG_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

# Initialize Career Agent with tools and memory store
agent = CareerAgent(tools, PG_URI)
agent.setup_memory_and_store()
agent.build()
graph = agent.get_graph()

# ======================== File & UI Utilities ========================

def extract_text_from_pdf(file_path):
    """Extract text content from uploaded PDF file."""
    if not file_path:
        return "No file uploaded!", gr.update(visible=True)
    
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return "\n".join([page.page_content for page in pages])  

def hide_component():
    """Hide Gradio component."""
    return gr.update(visible=False)

def show_component():
    """Show Gradio component."""
    return gr.update(visible=True)

def enable_button():
    return gr.update(interactive=True, variant="primary")

# ======================== State Checking Functions ========================

def get_jds(config):
    """Return the latest CV content from state."""
    jds = graph.get_state(config).values.get("jds", [])
    return jds

def get_reviewed_cv_text(config):
    """Return the latest CV content from state."""
    return graph.get_state(config).values.get("new_cv", "Not available")

def get_uploaded_cv_text(config):
    """Return the latest CV content from state."""
    return graph.get_state(config).values.get("cv", "Not available")

def get_thread_summary(config):
    """Return the summarized chat history from state."""
    return graph.get_state(config).values.get("chat_history_summary", "Not available")

def get_user_info_memory(config):
    """Retrieve long-term memory info of user stored across threads."""
    user_id = config["configurable"]["user_id"]
    namespace = ("user_info", user_id)
    try:
        user_info = graph.store.get(namespace, "info")
        if user_info:
            user_info = user_info.value
            print("user_info:", user_info)
            return user_info
        return "Not available"
    except Exception:
        return "Fail to get user info"

def refresh_internal_state(config):
    """Update UI with new CV, thread summary, and user memory."""
    return get_uploaded_cv_text(config), get_reviewed_cv_text(config), get_thread_summary(config), get_user_info_memory(config), get_jds(config)

# ======================== Config Initialization ========================

def initialize_config_and_ui(thread_id, user_id):
    """Load config, chat history, and CV info based on user/thread ID."""
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    state = graph.get_state(config).values
    chat_history = []

    if state:
        for mess in state["messages"]:
            if isinstance(mess, HumanMessage):
                if mess.content.endswith('/no_think'):
                    user_chat = mess.content[:-9]
                elif mess.content.endswith('/think'):
                    user_chat = mess.content[:-6]
                else:
                    user_chat = mess.content
                    
                chat_history.append({"role": "user", "content": user_chat, "metadata" : {"id": mess.id}})
            elif isinstance(mess, AIMessage):
                if mess.tool_calls and not mess.content: #call tool
                    for tool in mess.tool_calls:
                        chat_history.append({"role": "assistant", "content": f"{tool['args']}", "metadata": {"title": f"Calling {tool['name']}", "id": mess.id}})
                        
                else: #response
                    think_message, chat_message = split_message(mess.content)
                    if think_message:
                        chat_history.append({"role": "assistant", "content": "", "metadata": {"title": think_message, "id": mess.id}})
                    chat_history.append({"role": "assistant", "content": chat_message, "metadata": {"id": mess.id}})
                    
            elif isinstance(mess, ToolMessage):
                chat_history.append({"role": "assistant", "content": str(mess.content)[:100] + "...", "metadata": {"title": f"Considering tool {mess.name} response...", "id": mess.id}})
                
        return chat_history, config
    else:
        return "", config

# ======================== Chat Handling ========================

def handle_user_input(user_message, chat_history):
    """Add user text and uploaded file (if any) to chat history."""
    print("-u-")
        
    if user_message["files"]:
        file_content = extract_text_from_pdf(user_message["files"][0])
        return gr.MultimodalTextbox(value=None, interactive=False), chat_history + [
            {"role": "user", "content": user_message["text"], "metadata": {"id": str(uuid.uuid4())}},
            {"role": "user", "content": file_content, "metadata": {"title": "File included", "id": str(uuid.uuid4())}},
        ]
    else:
        return gr.MultimodalTextbox(value=None, interactive=False), chat_history + [
            {"role": "user", "content": user_message["text"], "metadata": {"id": str(uuid.uuid4())}},
        ]

# def stream_bot_response(config, chat_history):
#     """Bot responds to last message; streams token-by-token for animation."""
#     print("-b-")
#     try:
#         last_message = chat_history[-1]

#         # Incase user upload file
#         if last_message["metadata"].get("title", ""):
#             state = {
#                 "messages": [HumanMessage(chat_history[-2]["content"]+"\no_think", id=chat_history[-2]["metadata"]["id"])],
#                 "cv": last_message["content"],
#             }
#         else:
#             state = {
#                 "messages": [HumanMessage(last_message["content"]+"\no_think", id=last_message["metadata"]["id"])],
#             }

#         print(config)
        
#         # Prepare for new messages
#         chat_history.append({"role": "assistant", "content": "", "metadata": {"title": "",}})
        
#         THINK_FLAG = False
#         for i, (msg, metadata) in enumerate(graph.stream(state, config, stream_mode="messages")):
#             if metadata["langgraph_node"] == "agent":
#                 print(i, msg)
#                 if msg.tool_calls and not msg.content: #call tool
#                     chat_history[-1]["metadata"]["title"] = f"Calling {', '.join(tool['name'] for tool in msg.tool_calls)}"
                        
#                 else: #response
#                     new_token = msg.content
#                     if new_token.strip() == "<think>":
#                         print("start thinking token")                
#                         THINK_FLAG = True
#                         chat_history[-1]["metadata"]["title"] = ""
                        
#                     elif new_token.strip() == "</think>":
#                         print("end thinking token")                
#                         THINK_FLAG = False
#                         chat_history.append({"role": "assistant", "content": ""})
                        
#                     else:                    
#                         if THINK_FLAG:    
#                             chat_history[-1]["metadata"]["title"] += new_token
#                         else:
#                             if chat_history[-1]["metadata"]["title"] == f"Considering tool response...":
#                                 chat_history.append({"role": "assistant", "content": ""})
                                
#                             chat_history[-1]["content"] += new_token
                                    
                    
#                     yield chat_history
            
#             elif metadata["langgraph_node"] == "tools":
#                 chat_history[-1]["metadata"]["title"] = f"Considering tool response..."
            
#             else:
#                 pass
            
#             yield chat_history
    
#     except Exception as e:
#         chat_history.append({"role": "assistant", 
#                             "content": f"Got error: {e}"})
#         yield chat_history

import re
from typing import Dict

def split_message(text: str) -> str:
    # Tìm nội dung trong <think>...</think> ngay ở đầu
    match = re.match(r'<think>(.*?)</think>(.*)', text, re.DOTALL)
    # print("input mess------", text)
    
    if match:
        think_content = match.group(1).strip()
        outside_content = match.group(2).strip()
    else:
        # Nếu không tìm thấy <think> thì coi toàn bộ là outside
        think_content = ""
        outside_content = text.strip()
    # print("think------", think_content)
    # print("chat------", outside_content)
    if outside_content.startswith('<think>'):
        outside_content = outside_content[6:]
    
    return think_content, outside_content

            
def stream_bot_response(config, chat_history, think):
    """Bot responds to last message; streams token-by-token for animation."""
    print("-b-")
    
    print("think mode: ",think, "config", config)
    if think:
        add_in = " /no_think"
    else:
        add_in = " /think"
    
    print("chat_hist: ", chat_history)
    last_message = chat_history[-1]
    # print("lastmess", last_message)
    try:
        if last_message["metadata"].get("title", ""):
            state = {
                "messages": [HumanMessage(chat_history[-2]["content"]+add_in, id=chat_history[-2]["metadata"]["id"])],
                "cv": last_message["content"],
            }
        else:
            state = {
            "messages": [HumanMessage(last_message["content"]+add_in, id=last_message["metadata"]["id"])],
        }
    except Exception:
        state = {
            "messages": [HumanMessage(last_message["content"]+add_in, id=last_message["metadata"]["id"])],
        }
      
    #   ----------------------- Stream mode ------------------      
    for i, (msg, metadata) in enumerate(graph.stream(state, config, stream_mode="messages")):
        if metadata["langgraph_node"] == "agent":
            
            if chat_history[-1].get('metadata', {}).get('title', '')[:7] == "Waiting":                
                    chat_history = chat_history[:-1]
                    
            if msg.tool_calls and not msg.content: #call tool
                print(i, "agent call tool", msg)
                for tool in msg.tool_calls:
                    chat_history.append({"role": "assistant", "content": f"{tool['args']}", "metadata": {"title": f"Calling {tool['name']}", "id": msg.id}})
                    
            else: #response
                
                think_message, chat_message = split_message(msg.content)
                print(i, "agent message---")
                    
                if think_message:
                    print("i did think")
                    chat_history.append({"role": "assistant", "content": "", "metadata": {"title": think_message, "id": msg.id}})
                
                chat_history.append({"role": "assistant", "content": chat_message, "metadata": {"id": msg.id}})
                                            
        elif metadata["langgraph_node"] == "tools":
            print(i, "tools", msg)
            if chat_history[-1].get('metadata', {}).get('title', '')[:7] == "Waiting":
                    chat_history = chat_history[:-1]
            think_message, chat_message = split_message(msg.content)
                    
            chat_history.append({"role": "assistant", "content": chat_message, "metadata": {"title": f"Finish calling {msg.name}", "id": msg.id}})
        
        elif metadata["langgraph_node"] in ["filter_&_summarize_messages", "extract_user_info"]:
            if not chat_history[-1].get('metadata', {}).get('title', '')[:7] == "Waiting":
                chat_history.append({"role": "assistant", "content": "", "metadata": {"title": "Waiting"}})
                
            elif msg.response_metadata.get('done', False):
                chat_history.append({"role": "assistant", "content": "", "metadata": {"title": f"Finish updating memory", "id": msg.id}})
                
            else:
                # print("damn", msg)
                chat_history[-1]['metadata']['title'] = f"Waiting memory updates {'.'*(i%10)}"

        else:
            # print("damn", metadata["langgraph_node"], msg)
            
            if not chat_history[-1].get('metadata', {}).get('title', '')[:7] == "Waiting":
                chat_history.append({"role": "assistant", "content": "", "metadata": {"title": "Waiting"}})
            else:
                chat_history[-1]['metadata']['title'] = f"Waiting {metadata['langgraph_node']} {'.'*(i%10)}"
        
        yield chat_history
    print("end -b-")
    return chat_history
      # ----------------------- Stream mode :update ------------------      
        

        

def edit_message(history, edit_data: gr.EditData):
    """Edit a message in the chat UI."""
    new_history = history[:edit_data.index]
    msg_id = history[edit_data.index]["metadata"]["id"]
    new_history.append({"role": "user", "content": edit_data.value, "metadata": {"id": msg_id}})
    return new_history

def fork_message(config, chat_history):
    """Fork state from last message in chat and create a new config."""
    hist = graph.get_state_history(config)
    last_message = chat_history[-1]

    for i, s in enumerate(hist):
        if i % 2 == 0 and s.values["messages"][-1].id == last_message["metadata"]["id"]:
            to_fork = s.config
            break

    fork_config = graph.update_state(
        to_fork,
        {"messages": [HumanMessage(content=last_message["content"], id=last_message["metadata"]["id"])]},
    )
    fork_config["configurable"]["user_id"] = config["configurable"]["user_id"]
    return fork_config

def remove_checkpoint_from_config(config):
    """Remove checkpoint ID from config to reset tracking."""
    config["configurable"].pop("checkpoint_id", None)
    return config

# ======================== User/Thread Utilities ========================

def generate_new_id():
    """Generate a new unique user/thread ID."""
    print("---gen id---")
    return gr.update(value=str(uuid.uuid4()))

def update_user_id_dropdown(choices, new_id=False):
    """Update dropdown choices when a new user ID is added."""
    if not new_id:
        new_id = str(uuid.uuid4())
    if new_id not in choices:
        print("---add choice---")
        choices.append(new_id)
    return gr.update(value=new_id, choices=choices), choices

def insert_user_thread_to_db(user_id, thread_id):
    """Insert user-thread mapping to PostgreSQL DB."""
    print("---add db---")
    from psycopg import connect
    conn = connect(PG_URI, autocommit=True)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_thread (user_id, thread_id)
        VALUES (%s, %s)
        ON CONFLICT (user_id, thread_id) DO NOTHING
    """, (user_id, thread_id))

def get_or_create_user_thread(user_id):
    """Return user's existing thread or create a new one."""
    print("---check thread---")
    from psycopg import connect
    conn = connect(PG_URI, autocommit=True)
    cursor = conn.cursor()
    cursor.execute("""SELECT * FROM user_thread WHERE user_id = %s""", (user_id,))
    rows = cursor.fetchall()
    if rows:
        print(f"User id: {user_id} has {len(rows)} available thread")
        available_threads = [row[1] for row in rows]
        return gr.update(value=available_threads[0], choices=available_threads), available_threads
    else:
        print("New User")
        new_thread_id = str(uuid.uuid4())
        insert_user_thread_to_db(user_id, new_thread_id)
        return gr.update(value=new_thread_id, choices=[new_thread_id]), [new_thread_id]