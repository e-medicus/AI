from flask import Flask, render_template, request
from gpt_index import (
    SimpleDirectoryReader,
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
)
from langchain import OpenAI
import os

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = "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"


def createVectorIndex(path):
    max_input = 4096
    tokens = 256
    chunk_size = 600
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(
        max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size
    )

    # Define LLM
    llMPredictor = LLMPredictor(
        llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens)
    )

    # load data
    docs = SimpleDirectoryReader(path).load_data()

    # create vector Index
    vectorIndex = GPTSimpleVectorIndex(
        documents=docs, llm_predictor=llMPredictor, prompt_helper=prompt_helper
    )
    vectorIndex.save_to_disk("vectorIndex.json")

    return vectorIndex


vectorIndex = createVectorIndex("./data")


@app.route("/")
def home():
    return render_template("./index.html")


@app.route("/get")
def get_bot_response():
    user_text = request.args.get("msg")
    print(f"prompt: {user_text}")
    vIndex = GPTSimpleVectorIndex.load_from_disk("vectorIndex.json")
    response = vIndex.query(user_text, response_mode="compact")
    print(f"response: {response}")

    return str(response)


if __name__ == "__main__":
    app.run()
