from langchain_groq import ChatGroq
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")
search = GoogleSerperAPIWrapper()
tool =[search.run]

memory = MemorySaver()
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model=llm,
    tools=tool,
    system_prompt="you are a google search agent and you have to search for the user query and give answer in simple way for user",
    checkpointer=st.session_state.memory
)


print(memory)
st.subheader("Google Search Agent with quick answers with agent")

for massage in st.session_state.history:
    role = massage["role"]
    content = massage["content"]
    st.chat_message(role).markdown(content)
    
q = st.chat_input("Ask anything: ")
if q:
    st.chat_message("user").markdown(q)
    st.session_state.history.append({"role": "user", "content": q})
    res = agent.invoke(
    {"messages": [{"role": "user", "content": q}]},
    {"configurable":{
        "thread_id":"user123"
    }},
    stream_mode="messages"
                    )
    # answer = res["messages"][-1].content
    # st.chat_message("ai").markdown(answer)
    # st.session_state.history.append({"role": "ai", "content": answer})
    
    ai_container = st.chat_message("ai")        
    with ai_container:
        space = st.empty()
        msg = ""
        for chunk in res:
            # print(chunk)
            msg = msg + chunk[0].content
            space.write(msg)
        st.session_state.history.append({"role": "ai", "content": msg})

    
# print(res["messages"][-1].content[0]["text"])

