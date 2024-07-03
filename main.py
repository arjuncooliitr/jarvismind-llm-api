from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Load content from a URL, the entire post is actually worth reading
from langchain_community.document_loaders import WebBaseLoader
url = "https://github.com/adobe/aio-cli/blob/master/README.md"
loader = WebBaseLoader(url)
docs = loader.load()

from langchain_openai import AzureChatOpenAI
# Initialize the model with our deployment of Azure OpenAI
model = AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"])

# Create class with pydantic BaseModel
class SuggestionRequest(BaseModel):
    input_str: str

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

@app.post("/suggestaiocommand")
async def suggestCommand(request: SuggestionRequest):
    system_message = "Input is User Prompt. Your task as a CLI Assistant is to map user prompt to a command from the given content. The output should only be the mapped command. Do not output any command that is not in the content."
    human_message = "User prompt: " + request.input_str + ". Content: " + str(docs)

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message),
    ]

    parser = StrOutputParser()

    chain = model | parser
    # Let's pass the system and human message to the LLM API and invoke it
    output = chain.invoke(messages)
    return {"output": output}

# Run the server with uvicorn
# uvicorn main:app --reload
