import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from transformers import AutoTokenizer
from langchain.agents import AgentExecutor, ConversationalChatAgent
from langchain.tools import Tool
from clear_results import with_clear_container
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Set up memory
msgs = StreamlitChatMessageHistory(key="chat_message_history")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

st.set_page_config(page_title="RAG Docs Chatbot", page_icon="🔗")
st.title("RAG Docs Chatbot")

view_messages = st.expander("View the message contents in session state")
llm = Ollama(model="llama2")


def get_loader(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained("gpt2"), chunk_size=400, chunk_overlap=40
    )
    chunks = splitter.split_documents(text)
    return chunks


def get_vector_store(chunks):
    embeddings = OllamaEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_context_retriever(vector_store):
    # llm = Ollama(model="llama2")
    retriever = vector_store.as_retriever()
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    history_aware_retriever_chain = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever_chain


def get_conversation_rag_chain(history_aware_retriever_chain):
    # llm = Ollama(model="llama2")

    qa_system_prompt = """You are an assistant for question-answering and retrieval augmented tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know.\n\n\
        {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever_chain, document_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever(st.session_state.vector_store)
    conversation_rag_chain = get_conversation_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke(
        {
            "chat_history": st.session_state.chat_history,
            "input": user_input,
        }
    )
    if response["action"] == "ask_question":
        user_query = st.chat_input(response["question"])
        response = get_response(user_query)
    return response["answer"]


msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)
search = DuckDuckGoSearchAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="Use the DuckDuckGo search engine to find information",
    ),
    Tool(
        name="rag",
        func=get_response,
        description="Use the RAG model to retrieve relevant information from the URL",
    ),
]


def get_agent_response(user_input):
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)

    agent = AgentExecutor.from_agent_and_tools(
        tools=tools,
        agent=chat_agent,
        llm=llm,
        verbose=True,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    result = agent.invoke(input=user_input, context=memory.chat_memory)

    if "output" in result:
        return result["output"]
    else:
        return "\n".join(result["intermediate_steps"])


chat_container = st.container()
with st.sidebar:
    st.header("Setup")
    url = st.text_input("Enter URL to load documents")
    if url is None or url == "":
        st.info("Please enter a URL to load documents")

with chat_container:
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    if url is not None and url != "":
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am a bot. How can I help you?"),
            ]
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vector_store(
                chunk_text(get_loader(url))
            )

    user_query = st.chat_input("Type your message here")
    if user_query is not None and user_query != "":
        response = get_agent_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # Render current messages from StreamlitChatMessageHistory
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # If user inputs a new prompt, generate and draw a new response
    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        # Note: new messages are saved to history automatically by Langchain during run
        config = {"configurable": {"session_id": "any"}}
        response = get_agent_response.invoke(user_query)
        st.chat_message("ai").write(response.content)
