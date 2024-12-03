from langchain_core.pydantic_v1 import BaseModel
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
    keywords: List[str]                     # list of extracted key words from user
    key_importance: List[str]               # list of importance values for key words
    enrichment: List[str]                   # list of enrichment words
    enrichment_importance: List[str]        # list of importance values for enrichment words
    query: str                              # query generated
    response: str                           # name found based on queries
    quality: List[bool]                     # list of boolean for if each category in rubric has evaluation material       
    missing_info: List[str]                 # list of missing info categories
    missing_query: List[str]                # list of queries generated for missing info categories
    
# prompts

USER_PROMPT = """I want to find 10 speakers in Vancouver who would be interested in presenting at an AI safety event being hosted at UBC."""

KEYWORD_PROMPT = """You are a keyword specialist who is tasked with identifying the key ideas, concepts, 
and words associated with the user given prompt: """ + USER_PROMPT

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
{content}"""

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
Return the results in a structured format."""
ENRICHMENT_IMPORTANCE_PROMPT = ""
QUERY_PROMPT = """Using the extracted keywords, enriched keywords, and their importance, generate a search 
query. The query should focus on finding 10 speakers in Vancouver who are likely to open up internship 
positions for international students in business, technology, and science.
Keywords: {keywords}
Enriched Keywords: {enrichment}"""
QUALITY_PROMPT = ""
MISSING_INFO_PROMPT = ""

# set up tavily
tavily = TaviltyClient(api_key = os.environ["TAVILY_API_KEY"])

# set up ChatGPT
model = ChatOpenAI(model = "gpt-4o", temperature = 0)
        # temperature is currently at dummy value of 0

class Queries(BaseModel): #TODO update structure of responses for importance agent
    queries: List[str]

# agent implementation

def keyword_agent(state: AgentState):
    messages = [SystemMessage(content=KEYWORD_PROMPT)]
    try:
        response = model.invoke(messages)
        return {"keywords": response.content}
    except Exception as e:
        return {"error": str(e), "keywords": []}

def importance_agent(state: AgentState):  
    messages = [SystemMessage (content = KEYWORD_IMPORTANCE_PROMPT),
                HumanMessage (content = state['keywords'])]
    try:
        response = model.invoke(messages)
        return {"key_importance": response.content}
    except Exception as e:
        return {"error": str(e), "key_importance": []}

def enrichment_agent(state: AgentState):
    messages = [
        SystemMessage(content=ENRICHMENT_PROMPT.format(user_prompt=USER_PROMPT, keywords=", ".join(state.get("keywords", []))))
    ]
    try:
        response = model.invoke(messages)
        return {"enrichment": response.content.split(", ")}  # Assuming output is comma-separated
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