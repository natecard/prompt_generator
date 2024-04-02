import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

embeddings = OllamaEmbeddings()


def vectorstore_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    chunks = text_splitter.split_documents(docs)

    vector = FAISS.add_documents(chunks, embeddings)

    return vector


def context_retriever_chain(vector_store):
    llm = Ollama(model="llama2")

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_rag_chain(retriever_chain):
    llm = Ollama(model="llama2")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, document_chain)


def get_response(user_input):
    retriever_chain = context_retriever_chain(st.session_state.vector_store)
    conversation_chain = create_stuff_documents_chain(retriever_chain)
    response = conversation_chain.invoke(
        {"chat history": st.session_state.chat_history, "input": user_input}
    )
    return response["answer"]


st.set_page_config(page_title="LangChain", page_icon="🔗")
st.title("LangChain")

with st.sidebar:
    st.header("Setup")
    url = st.text_input("Enter URL to load documents")
    if url is None or url == "":
        st.info("Please enter a URL to load documents")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, how can I help you today?")
            ]
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = vectorstore_url(url)

        user_query = st.chat_input("Type your message here")
        if user_query is not None and user_query != "":
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)