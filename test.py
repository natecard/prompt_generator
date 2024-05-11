import getpass
import json
import logging
import os

from dotenv import load_dotenv
import streamlit as st
from transformers import AutoTokenizer
from typing import List

from langchain.agents import (
    AgentExecutor,
    create_json_chat_agent,
    ConversationalChatAgent
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.agents import AgentActionMessageLog, AgentFinish 
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import Tool, BaseTool


from Model_Specific_Prompts.midjourney_prompt_guide import midjourney_prompt
from Model_Specific_Prompts.gpt_4_prompt_guide import gpt_4_prompt

load_dotenv()
logging.basicConfig(level=logging.DEBUG)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
hf_token = os.environ.get("HF_TOKEN")

# Load the Ollama model and HF autotokenizer
llm = ChatOllama(model="llama3:8b-instruct-q8_0")
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token
)

# If the API key is not found in the environment, prompt the user to enter it
if not langsmith_api_key:
    langsmith_api_key = getpass("Enter your LangChain API key: ")
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

# Initiate streamlit page
st.set_page_config(page_title="Prompt Chatbot", page_icon="🔗")
st.title("Prompt Chatbot")


# Set up memory
msgs = StreamlitChatMessageHistory(key="chat_history")
if len(msgs.messages) == 0:
    st.session_state.steps = {}
    msgs.add_ai_message(
        "How can I help you? You can ask me questions or search for information."
    )

# Buffer for storing chat history using the StreamlitChatMessageHistory
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)


# Load the WebBaseLoader to load documents from a URL
def get_loader(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


# Tokenize and chunk the text using the RecursiveCharacterTextSplitter and the Gemma tokenizer
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(text)
    return chunks

# Initialize the OllamaEmbeddings and FAISS vector store
def get_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="llama3:8b-instruct-q8_0")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_context_retriever(vector_store):
    retriever = vector_store.as_retriever()
    # This should be able to retrieve the context from the chat history via the msgs.messages
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is.\n\n
        """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("ai", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            # ("human", "{input}"),
        ]
    )
    history_aware_retriever_chain = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever_chain


def get_conversation_rag_chain(history_aware_retriever_chain):

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            # ("ai", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever_chain, document_chain)


class Response(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    sources: List[int] = Field(
        description="List of page chunks that contain answer to the question. Only include a page chunk if it contains relevant information"
    )


def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )


class ConversationRAGTool(BaseTool):
    name = "conversation_rag"
    description = "Use the Conversation RAG chain to answer the query based on the provided context."

    def _run(self, query, chat_history):
        """Use the tool for answering queries."""
        result = conversation_rag_chain({"chat_history": chat_history, "input": query})
        return result["result"]

    def _arun(self, query, chat_history):
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")


def conversation_rag_tool_func(query, chat_history):
    """Wrapper function to call the ConversationRAGTool."""
    tooling = ConversationRAGTool()
    return tooling._run(query, chat_history)


# Set up the RAG tool
rag_tools = [
    Tool(
        name="rag",
        func=conversation_rag_tool_func,
        description="Utilize a RAG model to generate responses to user queries",
    ),
]


def run_conversation(url):
    if url is not None and url != "":
        # if "vector_store" not in st.session_state:
            # try:
                docs = get_loader(url)
                chunks = chunk_text(docs)
                st.session_state.vector_store = get_vector_store(chunks)

                history_aware_retriever_chain = get_context_retriever(
                        st.session_state.vector_store
                )
                conversation_rag_chain = get_conversation_rag_chain(
                        history_aware_retriever_chain
                )
                return conversation_rag_chain
            # except Exception as e:
                # st.error(f"Error loading documents: {e}")
        # else:
        #     url = "https://en.wikipedia.org/wiki/Main_Page"
        #     docs = get_loader(url)
        #     chunks = chunk_text(docs)
        #     st.session_state.vector_store = get_vector_store(chunks)


search_wrapper = DuckDuckGoSearchAPIWrapper(region="en-us", max_results=5)
# Set up the DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun(verbose=True, api_wrapper=search_wrapper)

# Set up the search tools list
search_tools = [
    Tool(
        name="Search",
        func=search_tool.run,
        description="Use the DuckDuckGo search engine to find information",
    ),
]


def format_agent_scratchpad(intermediate_steps):
    messages = []
    for step, output in intermediate_steps:
        if step.log_prefix:
            messages.append(AIMessage(content=f"{step.log_prefix}: {output}"))
        else:
            messages.append(HumanMessage(content=output))
    return messages


def get_search_response(user_input, llm, memory, target_model_name):
    """
    Retrieves a response for a given search query.

    Args:
        user_input (str): The search query provided by the user.
        llm: The language model used for generating the response.
        memory: The memory used by the conversational agent.
        target_model_name (str): The name of the target model.

    Returns:
        Dict[str]: The response generated by the conversational agent.
    """
    if target_model_name == "midjourney":
        search_tool_prompt = midjourney_prompt(user_input, search_tools, "Search")
    if target_model_name == "gpt_4":
        search_tool_prompt = gpt_4_prompt(user_input, search_tools, "Search")
        chat_agent = create_json_chat_agent(
            llm=llm, tools=search_tools, prompt=search_tool_prompt
        )

    agent = AgentExecutor(
        tools=search_tools,
        agent=chat_agent,
        llm=llm,
        verbose=False,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    try:
        result = agent.invoke(
            {
                "input": user_input,
                "chat_history": memory,
            }
        )
        if "Final Answer" in result:
            response = result["Final Answer"]
            return response
        if "no tool response" in result:
            retry = agent.invoke({"input": user_input, "chat_history": memory})
            return retry
        #     response = result["output"]
        # else:
        #     response = (
        #         "I'm sorry, I couldn't find a satisfactory answer to your search query."
        #     )

    # Hack fix to stop the error "Could not parse LLM output: `" from being raised
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix(
            "`"
        )
    return result

def get_regular_response(user_input, llm, memory):
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=rag_tools)

    agent = AgentExecutor.from_agent_and_tools(
        tools=rag_tools,
        agent=chat_agent,
        llm=llm,
        verbose=False,
        memory=memory,
        handle_parsing_errors=True,
    )

    try:
        result = agent.invoke(
            input=user_input,
            chat_history=memory,  
            tools=["rag"],
        )
        if "output" in result:
            response = result["output"]
        else:
            response = "I'm sorry, I couldn't find a satisfactory answer to your query."
    except Exception as e:
        logging.error(f"Error occurred while processing regular query: {e}")
        response = "I'm sorry, an error occurred while processing your query. Please try again later."

    return response



chat_container = st.container()
input_container = st.container()
with st.sidebar:
    st.header("Setup")
    url = st.text_input("Enter URL to load documents")
    if url is None or url == "":
        st.info("Please enter a URL to load documents")
    else:
        # Initialize the vector store and chains
        conversation_rag_chain = run_conversation(url)

with chat_container:
    target_option = st.selectbox(
        "Select the model you want the prompt to be optimized for...",
        (
            "GPT 4",
            "Llama 3",
            # "Stable Diffusion",
            # "Gemini",
            # "Midjourney",
            # "Cohere Command R Plus",
            # "Gemma",
            # "Mistral",
            # "DBRX",
            # "Qwen",
            # "Nous Research",
        ),
        # set default to GPT 4
        index=0,
        placeholder="Choose a model",
    )

    st.write("You selected:", target_option)
    target_model_name = target_option.lower().replace(" ", "_")

    # Display existing chat history
    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.status(
                    f"**{step[0].tool}**: {step[0].tool_input}", state="complete"
                ):
                    st.write(step[0].log)
                    st.write(step[1])
            st.write(msg.content)
    with input_container:
        # If user inputs a new prompt, generate and draw a new response
        user_query = st.chat_input(
            "Type your message here",
        )
    if user_query is not None and user_query != "":
        # Display user message in container
        # chat_container.chat_message("user").write(user_query)
        # if "search" in user_query.lower():
        #     response: Response = get_search_response(user_query, llm, memory)
        # else:
            # Use the initialized conversation_rag_chain

        with st.spinner("Generating response..."):
            with chat_container.chat_message("ai"):
                st_cb = StreamlitCallbackHandler(
                    st.container(), expand_new_thoughts=False
                )
                cfg = RunnableConfig()
                cfg["callbacks"] = [st_cb]
                response = get_regular_response(user_query, conversation_rag_chain, memory)
                # response = get_search_response(
                #     user_query, llm, memory, target_model_name=target_model_name
                # )
                st.write(response["output"])
                st.session_state.steps[str(len(msgs.messages) - 1)] = response[
                    "intermediate_steps"
                ]
