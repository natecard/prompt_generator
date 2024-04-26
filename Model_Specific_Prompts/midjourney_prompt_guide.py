from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


def midjourney_prompt(input, tools, tool_names):
    system = """Your goal is to improve the prompt given below: 
    --------------------
    Prompt: {input}
    --------------------
    Here are some good example prompts that you can use as a reference to improve the prompt given above:
    "Enchanted forest, lush vegetation, magical creatures, vibrant colors --ar 16:9 --q 2 --chaos 50 --no humans --style expressive"
    "Serene beach sunset, palm trees swaying, gentle waves, golden hour --ar 3:2 --q .5 --chaos 25 --no buildings --style scenic --tile"
    "Futuristic cityscape, neon lights, flying cars, dark atmosphere --aspect 1:1 --stylize 1000 --iw 1.5 --seed 42 --no birds --style 4c"
    --------------------
    Here are several tips on writing great image generation prompts:
    If the prompt is for image creation using Midjourney, then describe the artwork in great detail.
    If the image is a photograph or still frame photo then include the camera name, the resolution, if any adjustments should be present like exposure, filtering, focal length, the specific lens that is used to create the image.
    If it is a painting or computer generated image then include the style, the color palette, the brush strokes, the medium, the specific software that is used to create the image, or the era in which it was painted.
    If the image is a drawing or illustration then include the style, the color palette, the medium, the specific software that is used to create the image, or the era in which it was drawn.
    Include some of the following variables below, all negative prompts should be placed at the end of the prompt with the --no parameter: 
    --------------------
    Aspect Ratios
    --aspect, or --ar Change the aspect ratio of a generation.

    Chaos
    --chaos <number 0-100> Change how varied the results will be. Higher values produce more unusual and unexpected generations.

    Character Reference
    Use images as character references in your prompt to create images of the same character in different situations.

    Fast
    --fast override your current setting and run a single job using Fast Mode.

    Image Weight
    --iw <0-2> Sets image prompt weight relative to text weight. The default value is 1.

    No
    --no Negative prompting, --no plants would try to remove plants from the image.

    Quality
    --quality <.25, .5, or 1>, or --q <.25, .5, or 1> How much rendering quality time you want to spend. The default value is 1. Higher values use more GPU minutes; lower values use less.

    Random
    --style random, add a random 32 base styles Style Tuner code to your prompt. You can also use --style random-16, --style random-64 or --style random-128 to use random results from other lengths of Style Tuners.

    Relax
    --relax override your current setting and run a single job using Relax Mode.

    Repeat
    --repeat <1-40>, or --r <1-40> Create multiple Jobs from a single prompt. --repeat is useful for quickly rerunning a job multiple times.

    Seed
    --seed <integer between 0-4294967295> The Midjourney bot uses a seed number to create a field of visual noise, like television static, as a starting point to generate the initial image grids. Seed numbers are generated randomly for each image but can be specified with the --seed or --sameseed parameter. Using the same seed number and prompt will produce similar ending images.

    Stop
    --stop <integer between 10-100> Use the --stop parameter to finish a Job partway through the process. Stopping a Job at an earlier percentage can create blurrier, less detailed results.

    Style
    --style <raw> Switch between versions of the Midjourney Model Version 5.1 and 5.2.
    --style <4a, 4b, or 4c> Switch between versions of the Midjourney Model Version 4.
    --style <cute, expressive, original, or scenic> Switch between versions of the Niji Model Version 5.
    Use the /tune command to create a Style Tuner and generate custom style codes.

    Style Reference
    Use images as style references in your prompt to influence the style or aesthetic of images you want Midjourney to make.

    Stylize
    --stylize <number>, or --s <number> parameter influences how strongly Midjourney's default aesthetic style is applied to Jobs.

    Tile
    --tile parameter generates images that can be used as repeating tiles to create seamless patterns.

    Turbo
    --turbo override your current setting and run a single job using Turbo Mode.

    Video
    --video Saves a progress video of the initial image grid being generated. Emoji react to the completed image grid with ✉️ to trigger the video being sent to your direct messages. --video does not work when upscaling an image.

    Weird
    --weird <number 0-3000>, or --w <number 0-3000> Explore unusual aesthetics with the experimental --weird parameter.

    Word Choice
    Word choice matters. More specific synonyms work better in many circumstances. Instead of big, try tiny, huge, gigantic, enormous, or immense.

    Plural words and Collective Nouns
    Plural words leave a lot to chance. Try specific numbers. "Three cats" is more specific than "cats." Collective nouns also work, “flock of birds” instead of "birds.”

    Focus on What you Want
    It is better to describe what you want instead of what you don’t want. If you ask for a party with “no cake,” your image will probably include a cake. To ensure an object is not in the final image, try advanced prompting using thee --no parameter.

    Prompt Length and Details
    Prompts can be simple. A single word or emoji will work. However, short prompts rely on Midjourney’s default style, allowing it to fill in any unspecified details creatively. Include any element that is important to you in your prompt. Fewer details means more variety but less control.

    Try to be clear about any context or details that are important for the prompt. Think about:

    Subject: person, animal, character, location, object
    Medium: photo, painting, illustration, sculpture, doodle, tapestry
    Environment: indoors, outdoors, on the moon, underwater, in the city
    Lighting: soft, ambient, overcast, neon, studio lights
    Color: vibrant, muted, bright, monochromatic, colorful, black and white, pastel
    Mood: sedate, calm, raucous, energetic
    Composition: portrait, headshot, closeup, birds-eye view
    You should review best practices for the models mentioned in the input, you have access to the following tools to aid in your review of the best practices for the models mentioned in the input:

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
