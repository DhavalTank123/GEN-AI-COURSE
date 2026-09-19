from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_google_genai import ChatGoogleGenerativeAI

import streamlit as st

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=5)
# que = "who is gojo?"
# res = llm.invoke(que)
# print(res.text)

# while True:
#     que = input("User: ")
#     if que.lower() in ["exit", "quit"]:
#         print("Exiting the bot!!\n.")
#         break
#     res = llm.invoke(que)
#     print(res.text)

st.title("QnA Bot")
st.markdown("My Qna Bot With langchain and google genai.")

# streamlit run 1_qna_bot.py 

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


queary = st.chat_input("Ask me anything!")
if queary:
    st.session_state.messages.append({"role": "user", "content": queary})
    st.chat_message("user").markdown(queary)
    res = llm.invoke(queary)
    st.session_state.messages.append({"role": "ai", "content": res.text})
    st.chat_message("ai").markdown  (res.text)
    