# jarvismind-llm-api
This repo contains a python app that exposes an API for autosuggesting CLI commands using Langframe LLM integrations.

# Setup Instructions
1. Clone the repository into your local machine
2. Open the repository into your favourite editor
3. Install the required dependencies by running listed in `requirements.txt` by running `pip install -r requirements.txt`.
4. Create a `.env` file and add the required credentials and secrets there like AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION etc
5. Run the app : `uvicorn main:app --reload`
6. Access the Autosuggestion api,  sample curl:
    `curl --location --request POST 'http://localhost:8000/suggestaiocommand' \
    --header 'accept: application/json' \
    --header 'Content-Type: application/json' \
    --data-raw '{
    "input_str": "deploy app"
    }'`
