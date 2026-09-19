from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

SERPER_FALLBACK = ""
serper_key = os.getenv("SERPER_API_KEY") or SERPER_FALLBACK
search = GoogleSerperAPIWrapper(serper_api_key=serper_key)

gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not gemini_key:
    print("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment.\nPlease add it to your .env or export it before running.")
    raise SystemExit(1)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=gemini_key)
# result = search.run("who is india pm")
# print(result)

memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="you are a google search agent and you have to search for the user query and give answer in simple way for user",
    checkpointer=memory
)
# q = "find founders of google and right now who is ceo?"
# res = agent.invoke({"messages": [{"role": "user", "content": q}]})
# res["messages"][-1].content[0]["text"]

while True:
    q = input("Enter your query: ") 
    if q == "exit":
        break
    res = agent.invoke({"messages": [
        {"role": "user", "content": q}]},
        {"configurable": {
            "thread_id": "abcd123"
        }}
                       )
    print("Ai: ", res["messages"][-1].content[0]["text"])
