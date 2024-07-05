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
You are an AIO CLI command assistant. Use the following pieces of retrieved context to answer the question. Your task as a CLI Assistant is to map user prompt to the closest 2 commands in context and output only relevant mapped commands and brief description in the json format. Do not output any command that is not in the context.
Context: {context}
""")    

# Initialize the model with our deployment of Azure OpenAI
model = AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"])

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
    return {"output": rag_output}


# Run the server with uvicorn
# uvicorn main:app --reload
