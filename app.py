import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize retriever and LLM
embeddings = download_hugging_face_embeddings()
index_name = "chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# Streamlit UI

st.markdown("""
    <style>
        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-right: 3rem;
            padding-left: 3rem;
            padding-bottom: 2rem;
        }
        h1 {
            white-space: nowrap;
        }
    </style>
""", unsafe_allow_html=True)
st.set_page_config(page_title="🧠 Medical Chatbot", layout="centered")
st.title("🤖 Medical Chatbot (Groq + Pinecone)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("Type your medical question:", key="input")

if user_input:
    with st.spinner("Thinking..."):
        result = rag_chain.invoke({"input": user_input})
        bot_response = result["answer"]

        # Save interaction
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", bot_response))


chat_pairs = list(zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2]))
for user, bot in reversed(chat_pairs):
    user_msg = user[1]
    bot_msg = bot[1]

    st.markdown(
    f"""
    <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
        <div style="background-color: #58cc71; padding: 10px; border-radius: 15px 0px 15px 15px; max-width: 90%; width: fit-content;">
            <strong>👤 You:</strong> {user_msg}
        </div>
    </div>
    <div style="display: flex; justify-content: flex-start; margin-bottom: 20px;">
        <div style="background-color: #52acff; padding: 10px; border-radius: 0px 15px 15px 15px; max-width: 90%; width: fit-content;">
            <strong>🤖 Bot:</strong> {bot_msg}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

