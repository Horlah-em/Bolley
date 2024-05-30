import os
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import openai
import asyncio
from langchain_openai import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")




# Define the DuckDuckGo search tool
search = DuckDuckGoSearchRun()

def duck_wrapper(input_text):
    try:
        search_results = search.run(f"site:nba.com {input_text}")
        return search_results
    except ValueError as e:
        print(f"Error in duck_wrapper: {e}")
        return "Error in retrieving information."



# Define the function to handle general conversation using GPT-4
def openai_conversation_responder(query):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a friendly assistant."},
                  {"role": "user", "content": query}]
    )


# Define tools
tools = [
    Tool(
        name="Search NBA.com",
        func=duck_wrapper,
        description="useful to respond to user politely and answer sport related question especially on nba"
    ),
    Tool(
        name="OpenAI Conversation Responder",
        func=openai_conversation_responder,
        description="useful for responding to general conversation in a friendly manner"
    ),
]

# Set up the base template
template = """
You are an AI system designed to give both realtime and past updates on nba fixtures, club statistics and player statistics and also maybe makes predictions.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation)
Thought: Give a comprehensive summary from observation
Final Answer: observation

Remember you are a realtime data provider for nba fans.

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        history = kwargs.pop("history", "")
        kwargs["history"] = history
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"]
)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # If no match, return an AgentFinish with the current thought
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

llm = OpenAI(temperature=0.9)
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)
from langchain.memory import ConversationBufferMemory 
# For handling conversation history
memory = ConversationBufferMemory()  # No need to specify k as it defaults to None (unlimited)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory  # Pass the memory instance
)

async def async_run_llm_agent(query, history=None):
    try:
        output = await agent_executor.arun({"input": query, "history": history})
        return {"output": output}
    except Exception as e:
        print(f"Error in async_run_llm_agent: {e}")
        return {"output": "An error occurred while processing your request."}

# No need for another async function
async def run_llm_agent(query, history=None):
    return await async_run_llm_agent(query, history)