import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st

# load_dotenv() will load the environment variables from the .env file, where we have the API keys
load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

### TOOLS

# 1. Tool for searching on Google using Serper API

def search(query):
    url = "https://google.serper.dev/search"

    # convert the query to json format
    payload = json.dumps({
        "q": query
    })

    # The headers contain the API key and the content type in the request
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

# Testing with an example search query:
# search("what is the date of Burning Man 2024?")

# 2. Tool for scraping website using Browserless API. Combined with the summarization tool, it will summarize the content of the website if the content is too large but otherwise will return the content as it is.

def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the url to be scraped in the request body
    data = {
        "url": url
    }

    # Convert Python object to JSON string to be sent in the request body
    data_json = json.dumps(data)

    # Send the POST request with the browserless API key
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"

    # Here we send to browserlesss data_json instead of data, because the data needs to be in JSON format, along with the headers, and the url
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code. If beutifulsoup can't parse the content, it will return 200 but with a blank content.
    # Beutiful soup will get the text content of the website HTML page returned by the browserless API
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)

        # If the content is too large, summarize it with the summary function
        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

# Testing with an example url:
# scrape_website("what is langchain?", "https://python.langchain.com/en/latest/index.html")


# 3. Tool for Map reduce chain for summarizing text

def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    # The text splitter will split the text into chunks of 10000 characters with 500 characters overlap
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    
    # The text splitter will create a list of documents from the content
    docs = text_splitter.create_documents([content])

    # The map prompt will be used to generate the map prompt for each document chunk
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    # The map prompt template will be used to generate the map prompt for each document chunk
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    # The load_summarize_chain function will summarize all the text chunks and combine them into one summary
        # chain type is map_reduce, which means the map prompt will be used to generate the map prompt for each document chunk
        # you can have a differnt summarization prompt for the chunk or the total summary if you want. The map_prompt_template will be used to generate the map prompt for each document chunk, while the combine_prompt_template will be used to generate the combine prompt for the total summary
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    # The docs are the documents to be summarized and the objective is the objective & task that users give to the agent
    # This is where the map reduce chain will be run with the docs and objective
    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


# The classes below help the agent to run the tools above by defining the input schema for the tools

# First we have to help the agent by defining the input parameters for the tools
class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")

# Tnen we help the agent by defining when the tool should be run. The args_schema is the input parameters for the tool we difed above, and the _run function is the function that will be run when the tool is called

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "Useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    # We define what happens when the tool is called and also when the tool is called with the wrong input parameters
    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


## LANGCHAIN AGENT WITH TOOLS ABOVE

tools = [
    # Tool for searching on Google using Serper API. Because it is a single function, we are using a function to define the tool instead of a class
    Tool(
        name="Search",
        func=search,
        description="Useful for when you need to answer questions about current events, data. You should ask targeted questions."
    ),
    # Tool for scraping website using Browserless API. Combined with the summarization tool, it will summarize the content of the website if the content is too large but otherwise will return the content as it is. Because the tool has multiple functions, we are using a class to define the tool
    ScrapeWebsiteTool(),
]

# Note that duplicate repeating the rules is intentional, because we want to make sure the agent follows that rule
system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things I should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iterations
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, you should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, you should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            7/ You should not write anything that is not related to the objective"""
)

# The agent_kwargs is the input parameters for the agent. The extra_prompt_messages is the messages that will be shown to the agent before the agent starts to think. The system_message is the message that will be shown to the agent when the agent is thinking
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
# In the memory, we will store the conversation summary, which is the output of the agent. Recent exchanges are remembered word for word, and older exchanges are summarized.
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    # verbose means we will be able to see what the agent is thinking at each step
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


## STREAMLIT APP

# def main():
#     st.set_page_config(page_title="AI research agent", page_icon=":bird:")

#     st.header("AI research agent :bird:")
#     query = st.text_input("Research goal")

#     if query:
    
#         st.write("Doing research for ", query)

#         result = agent({"input": query})

#         st.info(result['output'])


# if __name__ == '__main__':
#     main()

## FASTAPI APP
# Set this as an API endpoint via FastAPI

app = FastAPI()

# The query is the input parameter for the agent
# FastAPI will automatically convert the incoming query to a JSON object and pass it to the agent

class Query(BaseModel):
    query: str

# When there is a post request to the API endpoint, the agent will be called with the query as the input parameter
@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content