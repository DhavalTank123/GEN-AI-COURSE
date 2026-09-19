from dotenv import load_dotenv

load_dotenv()
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import  add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing  import Annotated
import streamlit as st

class ChatState(BaseModel):
    messages:Annotated[list,add_messages]
    
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

def chatbotNode(state:ChatState) -> ChatState:
    res = llm.invoke(state.messages)
    state.messages = [res]
    return state    
memory = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node("Chatbot",chatbotNode)
graph.add_edge(START,"Chatbot")
graph.add_edge("Chatbot",END)

graph = graph.compile(checkpointer=memory)

# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())

config = {"configurable":{"thread_id":"my-bot-1"}}

while True:
    query = input("User: ")
    if query.lower() in ["quit", "exit", "bye"]:
        print("Thanks for using me !")
        break


    res = graph.invoke(
        {"messages":[{"role":"user", "content":query}]},
        config
        )

    ans = res["messages"][-1].text
    print("AI: ", ans)