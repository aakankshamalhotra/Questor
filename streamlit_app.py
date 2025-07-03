
# Questor.py — Your Personal Tutor with Gemini + FAISS
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\aakan\Downloads\gemini-tutor-project-bcefcf3b6e56.json"

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load environment variables (GEMINI_API_KEY)
load_dotenv()

# Load LLM (Gemini)
def load_llm():
    GOOGLE_API_KEY=os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        model_kwargs={"max_output_tokens": 6000}
    )
    return llm


# Load retriever (FAISS + Embeddings)
def load_retriever():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(
        "vectorstore/db_faiss",
        embedding,
        allow_dangerous_deserialization=True
        )
    retriever = db.as_retriever()
    return retriever


# Load the full QA chain
def get_qa_chain():
    llm = load_llm()
    retriever = load_retriever()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an expert tutor. Use the chat history and answer the new question in a detailed, step-by-step way with examples.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return qa_chain


# Streamlit UI

st.set_page_config(page_title="Questor Bot", page_icon="🤖")
st.title("🤖💡 Hi, I'm Questor. Your Personal AI Tutor with Gemini + RAG")

# Initialize messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

qa_chain = get_qa_chain()

# Show past messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Input box at the bottom
user_input = st.chat_input("Hello! Ask your question: ")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    response = qa_chain.invoke({
        "question": user_input,
        "chat_history": [
            m["content"] for m in st.session_state.messages if m["role"] == "user"
        ]
    })

    answer = response["answer"] if "answer" in response else str(response)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.chat_message("assistant").write(answer)

   


