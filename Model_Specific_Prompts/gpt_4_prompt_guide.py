from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def gpt_4_prompt(input, tools, tool_names):
    system = """
    First work out your own prompt to rephrase the input below. Then compare your prompt to the user's prompt and evaluate if the user's prompt is more suited or not. Don't decide if the user's prompt is correct until you have done the prompt yourself.
    Your goal is to improve the prompt given below: 
    --------------------
    Prompt: {input}
    --------------------

    You should review best practices for the GPT-4, you have access to the following tools to aid in your review of the best practices for the models mentioned in the input:

    {tools}

    The way you use the tools is by specifying a json blob.
    Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

    The only values that should be in the "action" field are: {tool_names}

    The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

    ```
    {{
    "action": $TOOL_NAME,
    "action_input": $INPUT
    }}
    ```

    ALWAYS use the following format:

    Question: the input prompt you must improve upon
    Thought: you should always think about what to do
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: the result of the action
    ... (this Thought/Action/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the best prompt for the original input prompt

    Begin! 
    
    Major Reminder to always use the exact characters `Final Answer` when responding.
    Now, improve the prompt. Remember to always use the exact characters 'Final Answer' when responding. Prompt: {input}
    """
    
    human = """

    RESPONSE FORMAT INSTRUCTIONS
    ----------------------------

    When responding to me, please output a response in one of two formats:

    **Option 1:**
    Use this if you want the human to use a tool.
    Markdown code snippet formatted in the following schema:

    json
    {{
        "action": string, \ The action to take. Must be one of {tool_names}
        "action_input": string \ The input to the action
    }}
    

    **Option #2:**
    Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

    json
    {{
        "action": "Final Answer",
        "action_input": string \ You should put the final prompt here
    }}
    

    USER'S INPUT
    --------------------
    Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

    {input}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    return prompt