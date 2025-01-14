import time 
import os 

from azure.search.documents import SearchClient
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


AZURE_OPENAI_API_KEY            = os.getenv('AZURE_OPENAI_API_KEY')
SEARCHAI_API_KEY                = os.getenv('SEARCHAI_API_KEY')
AZURE_SEARCH_ENDPOINT           = os.getenv('AZURE_SEARCH_ENDPOINT')
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT')
AZURE_OPENAI_GPT_ENDPOINT       = os.getenv('AZURE_OPENAI_GPT_ENDPOINT')
AZURE_OPENAI_GPT_DEPLOYMENT     = os.getenv('AZURE_OPENAI_GPT_DEPLOYMENT')
AZURE_OPENAI_GPT_API_VERSION    = os.getenv('AZURE_OPENAI_GPT_API_VERSION')
INDEX_NAME                      = os.getenv('INDEX_NAME')



openai_client_embeding = AzureOpenAI(
    api_key = AZURE_OPENAI_API_KEY,  
    api_version = "2023-05-15",
    azure_endpoint = AZURE_OPENAI_EMBEDDING_ENDPOINT
)

openai_client_gpt = AzureOpenAI(
    api_key = AZURE_OPENAI_API_KEY,  
    api_version = "2023-05-15",
    azure_endpoint = AZURE_OPENAI_GPT_ENDPOINT
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCHAI_API_KEY)
)


def generate_embeddings(text, model="text-embedding-ada-002"): 
    time.sleep(0.5)
    try:
        embeddings = openai_client_embeding.embeddings.create(input = [text], model=model).data[0].embedding
    except Exception as e:
        print(e)    
    return embeddings

GROUNDED_PROMPT="""
You are a friendly assistant that give information about database schema, relationship about other table and keys.
Answer the query using only the sources provided below in a friendly and concise manner.
Answer ONLY with the facts listed in the list of sources below.
If there isn't enough information below, say you don't know.
Do not generate answers that don't use the sources below.
Query: {query}
Sources:\n{sources}
"""

def search_result(query):
    embedding       = generate_embeddings(query)
    vector_query    = VectorizedQuery(vector=embedding, k_nearest_neighbors=3, fields="column_name_vector")

    search_results = search_client.search(
        search_text=query,
        vector_queries= [vector_query],
        include_total_count=True,
        top=10
    )

    sources_formatted = ""
    for document in search_results:
        data = ""
        column_name             = document["column_name"]
        table_name              = document["table_name"]
        definition              = document["definition"]
        description             = document["description"]
        referring_foreign_table = document["referring_foreign_table"]
        referring_foreign_key   = document["referring_foreign_key"]
        
        data = f"Column Name: {column_name}, Table Name: {table_name.replace('Table', '')}, "
        if definition:
            data = data + f"Definition: {definition}, "
        if description:
            data = data + f"Description: {description}, "
        if referring_foreign_table:
            data = data + f"Referring Foreign Table: {referring_foreign_table}, "
        if referring_foreign_key:
            data = data + f"Referring Foreign Key: {referring_foreign_key}"

        sources_formatted = sources_formatted + data 

    print(sources_formatted)
    print("\n.............\n")
    return sources_formatted


llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_GPT_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_GPT_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_GPT_API_VERSION,
    verbose=False,
    temperature=0.3,
)


GROUNDED_PROMPT = """
You are a friendly assistant that gives factual information about database table schemas, their relationships, and keys.
Answer the query concisely and only using the safe, factual sources provided below.
If the query or sources contain inappropriate content, politely state that the query cannot be answered.
{chat_history}
Query: {query}

Sources: {sources}
"""

memory = ConversationBufferMemory(memory_key="chat_history", input_key="query", return_messages=True,)
prompt_template = PromptTemplate(
        input_variables=["query", "chat_history", "sources"],
        template=GROUNDED_PROMPT
    )

chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)


def get_answer(query):
    sources = search_result(query)

    response = chain.predict(query = query, sources = sources)

    print(response)
    return response



