from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents import create_agent
import streamlit as st

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
    
if "agent" not in st.session_state:
    st.session_state.agent = None
   
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []


from dotenv import load_dotenv
load_dotenv()

def process_document(path):

    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()
    len(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 100)
    splittered_doc = splitter.split_documents(docs)

    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )
    vector_db = InMemoryVectorStore(embedding=embedding)
    vector_db.add_documents(splittered_doc)

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

    @tool
    def retrieve_context(query:str):
        """Retrieve documents relevant to a query from the knowledge base."""
        context = ""
        docs = vector_db.similarity_search(query=query, k=3)
        for doc in docs:
            context = doc.page_content + "\n\n"

        return context

    system_prompt = """You are a helpful assistant that answers questions using retrieved context. 
        My knowledge base consists of the details from the uploaded document. 
        ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."""

    memory = InMemorySaver()

    agent = create_agent(
        model = llm,
        tools = [retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )
    st.session_state.agent = agent
    st.session_state.document_uploaded =True

# while True:
#     query = input("User:")
#     if query.lower() == "Exit":
#         break
#     res = agent.invoke({"messages":[{"role":"user","content":query}]},{"configurable":{"thread_id":11}})
#     result = res["messages"][-1].text
#     print(result)


# upload ui

if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select pdf Files",type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing"):
            path = "./doc_files/"
            for file in uploaded:
                with open(path + file.name,"wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()
            
# chat ui

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)
        
    
    query = st.chat_input("Ask anything related to uploaded document")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user:").markdown(query)
        res = st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":11}}
        )
        result = res["messages"][-1].text

        st.chat_message("ai").markdown(result)
        st.session_state.messages.append({"role":"ai","content":result})