from langchain.prompts import PromptTemplate


def prompt_template(task, input, agent_scratchpad=""):
    template = f"""Your goal is to improve the prompt given below for {task} : 
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
Now, improve the prompt. Question: {input}

Are followup questions needed here: {agent_scratchpad}"""
    input_variables = ["input", "agent_scratchpad"]
    prompt = PromptTemplate(template, input_variables)
    return prompt
