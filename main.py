from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Create class with pydantic BaseModel
class SuggestionRequest(BaseModel):
    input_str: str


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Load, chunk and index the contents of the aio-cli readme.
loader = UnstructuredMarkdownLoader("./aiocliREADME.md")
docs = loader.load()

# Split the content into manageable chunks for better retrieval.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed the chunks and store them in ChromaDB for efficient retrieval.
vectorstore = Chroma.from_documents(documents=splits, embedding=AzureOpenAIEmbeddings(azure_deployment=os.environ["AZURE_EMBEDDINGS_DEPLOYMENT"]))


from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Set up the RAG chain for retrieving and generating answers.
retriever = vectorstore.as_retriever()
system_prompt = ("""
You are an AIO CLI command assistant. Use the following pieces of retrieved context to answer the question. Your task as a CLI Assistant is to map the user prompt to the closest 2 commands in the context. Output only the relevant mapped commands and brief descriptions in the following JSON schema:

    'commands': [
        
            'command': 'command1',
            'description': 'description1'
        ,
        
            'command': 'command2',
            'description': 'description2'
        
    ]


Do not output any command that is not in the context.
Context: {context}
""")    

# Initialize the model with our deployment of Azure OpenAI
model = AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"])

import json
import re

def extract_json_from_string(input_string):
    # Use regular expressions to find the JSON part of the string
    json_match = re.search(r'```json\n(.+)\n```', input_string, re.DOTALL)
    
    if json_match:
        json_string = json_match.group(1)
        # Parse the JSON string
        json_data = json.loads(json_string)
        return json_data
    else:
        raise ValueError("No JSON found in the input string")


@app.post("/suggestaiocommand")
async def suggestCommand(request: SuggestionRequest):

    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input_str}"),
        ]
    )

    rag_chain = (
    {"context": retriever | format_docs, "input_str": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

    # Let's pass the system and human message to the RAG API and invoke it
    rag_output = rag_chain.invoke(request.input_str)
    #return rag_output
    return extract_json_from_string(rag_output)


# Run the server with uvicorn
# uvicorn main:app --reload
