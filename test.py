import os
import getpass
from dotenv import load_dotenv
import streamlit as st

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.agents import AgentExecutor, ConversationalChatAgent
from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_core.messages import HumanMessage, AIMessage

from transformers import AutoTokenizer

from clear_results import with_clear_container

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
hf_token = os.environ.get("HF_TOKEN")

# If the API key is not found in the environment, prompt the user to enter it
if not langsmith_api_key:
    langsmith_api_key = getpass("Enter your LangChain API key: ")
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

# Initiate streamlit page
st.set_page_config(page_title="RAG Docs Chatbot", page_icon="🔗")
st.title("RAG Docs Chatbot")

# Set up memory
msgs = StreamlitChatMessageHistory(key="chat_message_history")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")
view_messages = st.expander("View Chat History")

# Load the Ollama model and HF autotokenizer
llm = Ollama(model="gemma")
gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", token=hf_token)


def get_loader(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        gemma_tokenizer,
        chunk_size=400,
        chunk_overlap=40,
    )
    chunks = splitter.split_documents(text)
    return chunks


def get_vector_store(chunks):
    embeddings = OllamaEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)


def get_context_retriever(vector_store):
    llm = Ollama(model="gemma")
    retriever = vector_store.as_retriever()
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("ai", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever_chain = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever_chain


def get_conversation_rag_chain(history_aware_retriever_chain):
    llm = Ollama(model="gemma")

    qa_system_prompt = """You are an assistant for question-answering and retrieval augmented tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know.\n\n\
        {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("ai", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever_chain, document_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever(st.session_state.vector_store)
    conversation_rag_chain = get_conversation_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke(
        {
            "chat_history": msgs.messages,
            "input": user_input,
        }
    )
    if response["action"] == "ask_question":
        user_query = st.chat_input(response["question"])
        response = get_response(user_query)
    return response["answer"]


search = DuckDuckGoSearchAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="Use the DuckDuckGo search engine to find information",
    ),
]


def get_agent_response(user_input):
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)

    agent = AgentExecutor.from_agent_and_tools(
        tools=tools,
        agent=chat_agent,
        llm=llm,
        verbose=False,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    # Check if the user input contains a specific keyword or pattern that indicates a web search is needed
    if "search" in user_input.lower():
        result = agent.run(
            user_input, search_quality_reflection=True, search_quality_score_threshold=4
        )
    else:
        result = agent.invoke(
            input=user_input,
            context=st.session_state.vector_store,
            tools=["rag"],
        )

    if "output" in result.lower():
        response = result["output"]
        print(f"Response from get_agent_response: {response}")  # Add this line
        return response
    else:
        response = "\n".join(result["intermediate_steps"])
        print(f"Response from get_agent_response: {response}")  # Add this line
        return response


chat_container = st.container()
with st.sidebar:
    st.header("Setup")
    url = st.text_input("Enter URL to load documents")
    if url is None or url == "":
        st.info("Please enter a URL to load documents")

with chat_container:
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    if url is not None and url != "":
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vector_store(
                chunk_text(get_loader(url))
            )

    # Render messages from StreamlitChatMessageHistory
    for msg in msgs.messages:
        if msg.type == "human":
            with st.chat_message("Human"):
                st.write(msg.content)
        elif msg.type == "ai":
            with st.chat_message("AI"):
                st.write(msg.content)
                print(f"Rendered message: {msg.content}")  # Add this line

    # If user inputs a new prompt, generate and draw a new response
    user_query = st.chat_input(
        "Type your message here", on_submit=with_clear_container, args=(True,)
    )
    if user_query is not None and user_query != "":
        # msgs.add_human_message(user_query)  # Add user message to chat history
        with st.spinner("Generating response..."):
            response = get_agent_response(user_query)
        with st.chat_message("AI"):
            st.write(response)
        print(f"Response before adding to chat history: {response}")  # Add this line
        with st.chat_message("assistant"):
            st.write(response)
        print(f"Response after adding to chat history: {response}")  # Add this line
        msgs.add_ai_message(response)  # Add AI response to chat history
