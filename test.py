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
    ConversationalChatAgent,
    AgentOutputParser,
    create_json_chat_agent,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import Tool, BaseTool


from prompt import prompt_template
from clear_results import with_clear_container
from Model_Specific_Prompts.midjourney_prompt_guide import midjourney_prompt

load_dotenv()
logging.basicConfig(level=logging.DEBUG)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
hf_token = os.environ.get("HF_TOKEN")

# Load the Ollama model and HF autotokenizer
llm = ChatOllama(model="qwen:14b")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B", token=hf_token)

# If the API key is not found in the environment, prompt the user to enter it
if not langsmith_api_key:
    langsmith_api_key = getpass("Enter your LangChain API key: ")
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

# Initiate streamlit page
st.set_page_config(page_title="RAG Chatbot", page_icon="🔗")
st.title("RAG Chatbot")


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
    embeddings = OllamaEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory="VDB_DIR"
    )
    vector_store.persist()
    return vector_store


def get_context_retriever(vector_store):
    retriever = vector_store.as_retriever()
    # This should be able to retrieve the context from the chat history via the msgs.messages
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is.\n\n\
        {msgs.messages}"""

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
    # qa_system_prompt = """You are an assistant for question-answering and retrieval augmented tasks. \
    #     Use the following pieces of retrieved context to answer the question. \
    #     If you don't know the answer, just say that you don't know.\n\n\
    #     """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            # ("ai", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(
        retriever=history_aware_retriever_chain, combine_docs_chain=document_chain
    )


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


def run_conversation(url=None):
    if "vector_store" not in st.session_state:
        if url is not None and url != "":
            docs = get_loader(url)
            chunks = chunk_text(docs)
            st.session_state.vector_store = get_vector_store(chunks)
        # history_aware_retriever_chain = get_context_retriever(
        #     st.session_state.vector_store
        # )
        # conversation_rag_chain = get_conversation_rag_chain(
        #     history_aware_retriever_chain
        # )
        # return conversation_rag_chain
    else:
        url = "https://en.wikipedia.org/wiki/Main_Page"
        docs = get_loader(url)
        chunks = chunk_text(docs)
        st.session_state.vector_store = get_vector_store(chunks)


# Set a reasonable limit for recursion depth
MAX_RECURSION_DEPTH = 5


# def get_response(user_input, recursion_depth=0, url=None):
# """
# Generates a response based on the user input using the conversation RAG model.

# Args:
#     user_input (str): The user's input.
#     recursion_depth (int, optional): The recursion depth to prevent infinite loops. Defaults to 0.

# Returns:
#     str: The generated response.

# Raises:
#     None

# Notes:
#     - If the recursion depth exceeds the limit, a default error message is returned.
#     - The function uses the conversation RAG model to generate a response based on the user input.
#     - If the model asks for clarification or a follow-up question, the user is prompted for input.
#     - If the user input indicates a search intent, search results are returned.
#     - If no specific intent is detected, the model's response is returned.
# """
# run_conversation(url)
# # Check if the recursion depth has exceeded the limit
# if recursion_depth >= MAX_RECURSION_DEPTH:
#     return "I'm sorry, I'm having trouble understanding your query. Could you please rephrase it or provide more context?"

# # Get the context retriever chain and conversation RAG chain
# retriever_chain = get_context_retriever(st.session_state.vector_store)
# conversation_rag_chain = get_conversation_rag_chain(retriever_chain)

# # Generate response using the conversation RAG model
# response = conversation_rag_chain.invoke(
#     {
#         # "chat_history": msgs.messages,
#         "input": user_input,
#     }
# )
# Uncomment if using the chain directly
# output_parser = AgentOutputParser()
# chain = retriever_chain | conversation_rag_chain | user_input | llm | output_parser
# chain.invoke()

# Check if the model is asking for clarification or a follow-up question
# if "ask_question" in response:
#     follow_up_question = response["ask_question"]
#     user_query = st.chat_input(follow_up_question)
#     question_response = get_response(user_query, recursion_depth + 1)
#     return AgentOutputParser(question_response).abatch()

# Check if the user input indicates a search intent
# else:
#     search_results = get_search_response(user_input, llm, memory)
#     return search_results

# # If no specific intent is detected, return the model's response
# else:
#     regular_response = get_regular_response(user_input, llm, memory)
#     return AgentOutputParser(regular_response).abatch()

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

    Returns:
        Dict[str]: The response generated by the conversational agent.
    """
    if target_model_name == "midjourney":
        search_tool_prompt = midjourney_prompt(user_input, search_tools, "Search")
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

    # def get_regular_response(user_input, llm, memory):
    # """
    # Get a regular response from the chat agent.

    # Args:
    #     user_input (str): The user's input/query.
    #     llm: The language model used by the chat agent.
    #     memory: The memory object used to store chat history.

    # Returns:
    #     str: The response generated by the chat agent.
    # """
    # chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=rag_tools)

    # agent = AgentExecutor.from_agent_and_tools(
    #     tools=rag_tools,
    #     agent=chat_agent,
    #     llm=llm,
    #     verbose=False,
    #     memory=memory,
    #     return_intermediate_steps=True,
    #     handle_parsing_errors=True,
    #     max_iterations=100,
    #     max_execution_time=60,
    #     output_parser=StrOutputParser(),
    # )

    # try:
    #     input_data = {"chat_history": memory.chat_memory.messages, "input": user_input}
    #     result = agent.run(input=input_data, tools=rag_tools)
    #     response = result["output"]
    # except Exception as e:
    #     logging.error(f"Error occurred while processing regular query: {e}")
    #     response = "I'm sorry, an error occurred while processing your query. Please try again later."

    # return response


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
            "Midjourney",
            "Stable Diffusion",
            "Llama 2",
            "Gemini",
            # "Cohere Command R Plus",
            # "Gemma",
            # "Mistral",
            # "DBRX",
            # "Qwen",
            # "Nous Research",
        ),
        index=1,
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
        chat_container.chat_message("user").write(user_query)
        # if "search" in user_query.lower():
        # response: Response = get_search_response(user_query, llm, memory)
        # else:
        #     # Use the initialized conversation_rag_chain
        #     response = get_regular_response(user_query, conversation_rag_chain, memory)

        with st.spinner("Generating response..."):
            with chat_container.chat_message("ai"):
                st_cb = StreamlitCallbackHandler(
                    st.container(), expand_new_thoughts=False
                )
                cfg = RunnableConfig()
                cfg["callbacks"] = [st_cb]
                response = get_search_response(
                    user_query, llm, memory, target_model_name=target_model_name
                )
                st.write(response["output"])
                st.session_state.steps[str(len(msgs.messages) - 1)] = response[
                    "intermediate_steps"
                ]
