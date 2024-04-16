from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


def prompt_template(input, tools, tool_names):
    system = """Your goal is to improve the prompt given below: 
    --------------------
    Prompt: {input}
    --------------------
    Here are several tips on writing great prompts:
    ------- 
    Start the prompt by stating that it is an expert in the subject. Put instructions at the beginning of the prompt and use ### or to separate the instruction and context Be specific, descriptive and as detailed as possible about the desired context, outcome, length, format, style, etc 
    ---------
    Here's an example of a great prompt:
    As a master YouTube content creator, develop an engaging script that revolves around the theme of "Exploring Ancient Ruins." Your script should encompass exciting discoveries, historical insights, and a sense of adventure. Include a mix of on-screen narration, engaging visuals, and possibly interactions with co-hosts or experts. The script should ideally result in a video of around 10-15 minutes, providing viewers with a captivating journey through the secrets of the past.
    Example: "Welcome back, fellow history enthusiasts, to our channel! Today, we embark on a thrilling expedition..."
    -----
    You have access to the following tools:

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

    Begin! Reminder to always use the exact characters `Final Answer` when responding.
    Now, improve the prompt. Prompt: {input}
    """

    human = """

    RESPONSE FORMAT INSTRUCTIONS
    ----------------------------

    When responding to me, please output a response in one of two formats:

    **Option 1:**
    Use this if you want the human to use a tool.
    Markdown code snippet formatted in the following schema:

    ```json
    {{
        "action": string, \ The action to take. Must be one of {tool_names}
        "action_input": string \ The input to the action
    }}
    ```

    **Option #2:**
    Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

    ```json
    {{
        "action": "Final Answer",
        "action_input": string \ You should put the final prompt here
    }}
    ```

    USER'S INPUT
    --------------------
    Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

    {input}"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    return prompt
