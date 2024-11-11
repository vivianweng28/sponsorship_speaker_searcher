from langchain_core.pydantic_v1 import BaseModel
from tavily import TaviltyClient
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from typing import TypedDict, List

# define agent state
class AgentState(TypedDict):
    keywords: List[str]                    # list of extracted key words from user
    key_importance: List[str]               # list of importance values for key words
    enrichment: List[str]                   # list of enrichment words
    enrichment_importance: List[str]        # list of importance values for enrichment words
    query: str                              # query generated
    response: str                           # name found based on queries
    quality: List[bool]                     # list of boolean for if each category in rubric has evaluation material       
    missing_info: List[str]                 # list of missing info categories
    missing_query: List[str]                # list of queries generated for missing info categories
    
# prompts

KEYWORD_PROMPT = """You are a keyword specialist who is tasked with identifying the key ideas, concepts, and words \
    associated with the user given prompt:

------

{content}"""

KEYWORD_IMPORTANCE_PROMPT = ""
ENRICHMENT_PROMPT = ""
ENRICHMENT_IMPORTANCE_PROMPT = ""
QUERY_PROMPT = ""
QUALITY_PROMPT = ""
MISSING_INFO_PROMPT = ""

# set up tavily
tavily = TaviltyClient(api_key = os.environ["TAVILY_API_KEY"])

# set up Gemini
model = ChatGoogleGenerativeAI(model = "gemini-ultra", temperature = 0)
        # temperature is currently at dummy value of 0

class Queries(BaseModel):
    queries: List[str]

# keyword node
def keyword_node(state: AgentState) {
    messages = [SystemMessage(content = KEYWORD_PROMPT)]
    response = model.invoke(messages)
    return {"keywords" : response.content}
}

#importance node
def importance_agent(state: AgentState) {
    
}