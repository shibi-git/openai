import os
from flask import Flask, request, Response, jsonify, render_template, stream_with_context
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

app = Flask(__name__)

# Use managed identity in Azure (fallback to env vars locally)
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

if os.getenv("AZURE_OPENAI_API_KEY"):
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-08-01-preview"
    )
else:
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-08-01-preview"
    )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    messages = request.json.get("messages", [])
    
    def event_stream():
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.7,
            stream=True
        )
        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content or ""
                yield content

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")