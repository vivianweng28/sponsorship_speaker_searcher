from langchain_core.pydantic_v1 import BaseModel
from enum import Enum
from tavily import TaviltyClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os 
from typing import TypedDict, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

memory = SqliteSaver.from_conn_string(":memory:")

# define agent state
class AgentState(TypedDict):
    type: Enum                              # target type of query
    keywords: List[str]                     # list of extracted key words from user
    key_importance: List[str]               # list of importance values for key words
    enrichment: List[str]                   # list of enrichment words
    enrichment_importance: List[str]        # list of importance values for enrichment words
    query: str                              # query generated
    name: str                               # name found based on queries
    evidence: List[str]                     # name found based on queries
    quality: List[bool]                     # list of boolean for if each category in rubric has evaluation material       
    missing_info: List[str]                 # list of missing info categories
    missing_query: List[str]                # list of queries generated for missing info categories

# user prompt

USER_PROMPT = """I want to find speakers in Vancouver who would be interested in presenting at an AI safety event being hosted at UBC."""

# type identification
class Type(Enum):
    Person = 1 
    Company = 2


# LLM prompts

IDENTIFICATION_PROMPT = """Given this prompt: {user_prompt}, identify if the prompt is asking to identify a 
person or a company by responding with 1 for person, 2 for company"""

KEYWORD_PROMPT = """You are a keyword specialist who is tasked with identifying the key ideas, concepts, 
and words associated with the user given prompt: {user_prompt}. The format of the response should be a list separated
by commas."""

KEYWORD_IMPORTANCE_PROMPT = """You are an expert at evaluating the importance of words in relation to a 
given topic or prompt. Your task is to assign an importance value to each word in a list, based on how 
strongly the word relates to the topic described in the given prompt. Use a scale of 1 to 5, where:

5: The word is highly critical and central to the topic.
4: The word is very important and frequently associated with the topic.
3: The word is somewhat important but less directly connected to the topic.
2: The word has minor relevance and is only tangentially related.
1: The word is not relevant to the topic.

Here is the prompt: """ + USER_PROMPT + """. Here is the list of words: 
-------
{content}.
Format the results in a list of the scores that match the order of words given, with the scores separated by a comma.
"""

ENRICHMENT_PROMPT = """
You are an advanced language model tasked with generating enrichment words to enhance the depth and variety of a provided prompt. 
Enrichment words are closely related terms, synonyms, or associated phrases that can help make the input more nuanced, diverse, and expressive.

Instructions:

You will be provided with:

A main prompt to guide the context.
A list of keywords with their relative importance on a scale from 1 (low) to 5 (high).
For each keyword, generate a list of 5-7 enrichment words or phrases that:

Reflect the keyword's core meaning or associations.
Respect its relative importance (e.g., prioritize highly important keywords with broader or more nuanced enrichment words).
Stay relevant to the main prompt.
Return the results in a format where all generated context terms are in a list, with each generated context term 
separated by a comma."""
ENRICHMENT_IMPORTANCE_PROMPT = """You are an expert at evaluating the importance of context words in relation to a 
given topic or prompt. Your task is to assign an importance value to each word in a list, based on how 
strongly the word relates to the topic described in the given prompt and a list of words that were ranked similarly.  Use a scale of 1 to 5, where:

5: The word is highly critical and central to the topic.
4: The word is very important and frequently associated with the topic.
3: The word is somewhat important but less directly connected to the topic.
2: The word has minor relevance and is only tangentially related.
1: The word is not relevant to the topic.

Here is the prompt: """ + USER_PROMPT + """. Here is the list of words and their corresponding importance: 
{keywords} {key_importance}. 
Here is the list of words to grade: 
{enrichment}"""
QUERY_PROMPT = """Using the extracted keywords, enriched keywords, their importance and the corresponding given prommpt, 
generate a search query. 
Keywords: {keywords}
Enriched Keywords: {enrichment}

Format the results in a list of the scores that match the order of enriched keywords given, with the scores separated by a comma.
"""
QUALITY_PROMPT = """"""
MISSING_INFO_PROMPT = ""

# set up tavily
tavily = TaviltyClient(api_key = os.environ["TAVILY_API_KEY"])

# set up ChatGPT
model = ChatOpenAI(model = "gpt-4o", temperature = 0)
        # temperature is currently at dummy value of 0

class Queries(BaseModel): #TODO update structure of responses for importance agent
    queries: List[str]

# agent implementation

def identification_agent(state: AgentState): 
    messages = [SystemMessage(content = IDENTIFICATION_PROMPT.format(user_prompt = USER_PROMPT))]
    try:
        response = model.invoke(messages)
        return {"type: response.content"}
    except Exception as e:
        return {"error": str(e), "type": []}

def keyword_agent(state: AgentState):
    messages = [SystemMessage(content=KEYWORD_PROMPT)]
    try:
        response = model.invoke(messages)
        keywords = response.content.split(", ")  # Assuming output is comma-separated
        state["keywords"] = keywords
        return {"keywords": keywords}
    except Exception as e:
        return {"error": str(e), "keywords": []}

def importance_agent(state: AgentState):
    messages = [
        SystemMessage(content=KEYWORD_IMPORTANCE_PROMPT.format(
            content=", ".join(state.get("keywords", []))
        ))
    ]
    try:
        response = model.invoke(messages)
        importance = [int(x.strip()) for x in response.content.split(",")]  
        state["key_importance"] = importance
        return {"key_importance": importance}
    except Exception as e:
        return {"error": str(e), "key_importance": []}

def enrichment_agent(state: AgentState):
    messages = [
        SystemMessage(content=ENRICHMENT_PROMPT.format(
            keywords=", ".join(state.get("keywords", []))
        ))
    ]
    try:
        response = model.invoke(messages)
        enrichment = response.content.split(", ") 
        state["enrichment"] = enrichment
        return {"enrichment": enrichment}
    except Exception as e:
        return {"error": str(e), "enrichment": []}

def enrichment_importance_agent(state: AgentState):
    messages = [
        SystemMessage(content=ENRICHMENT_IMPORTANCE_PROMPT.format(content=", ".join(state.get("enrichment", []))))
    ]
    try:
        response = model.invoke(messages)
        return {"enrichment_importance": response.content}
    except Exception as e:
        return {"error": str(e), "enrichment_importance": []}
    
def query_agent(state: AgentState):
    messages = [
        SystemMessage(content = QUERY_PROMPT)
    ]
    try:
        response = model.invoke(messages)
        return {"query": response.content}
    except Exception as e:
        return {"error": str(e), "enrichment_importance": []}
