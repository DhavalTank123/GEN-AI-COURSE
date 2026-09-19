from dotenv import load_dotenv
load_dotenv()


# db,llm,tool,memory,agent,system

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver  
from langchain.agents import create_agent
import streamlit as st


db = SQLDatabase.from_uri("sqlite:///task.db")

# db.run("""
#        CREATE TABLE IF NOT EXISTS tasks(
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         title TEXT NOT NULL,
#         desc TEXT,
#         status TEXT CHECK(status IN ('pending','In progress','complated')) DEFAULT 'pending',
#         created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#     """)

# print("DB table Created ")

system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table. 

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser."

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, dec, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, dec, status(pending/in progress/completed), created.
"""


llm = ChatGroq(model="openai/gpt-oss-120b")
toolkit = SQLDatabaseToolkit(db = db,llm = llm)
tools = toolkit.get_tools()

# for tool in tools:
#     print(tool)

# memory = InMemorySaver()

@st.cache_resource
def get_agent():
    agent =  create_agent(
    model  = llm,
    tools = tools,
    checkpointer = InMemorySaver(),
    system_prompt=system_prompt
    )
    return agent

agent = get_agent()

st.subheader("Task Ai Assistant Manager")
if "message" not in st.session_state:
    st.session_state.messages = []

for massage in st.session_state.messages:
    role = massage["role"]
    content = massage["content"]
    st.chat_message(role).markdown(content)

    

prompt = st.chat_input("Ask me your task")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("ai"):
        with st.spinner("Processing..."): 
        
            res = agent.invoke({"messages":{"role":"user","content":prompt}},
                        {"configurable":{"thread_id":"1"}})
    
            result = res["messages"][-1].content
            st.markdown(result)
            st.session_state.messages.append({"role":"ai","content":result})


  
