import os
import sys
testdir = os.path.dirname(__file__)
srcdir = "./backend"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from flask import Flask
from requests import request
app = Flask(__name__)

from backend import llm
from backend import recsys


llm = llm.ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=5,
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)

sampler_map = {
    recsys.SamplerType.GNEWS: recsys.GNewsSampler(),
}

@app.route("/generate_topics", methods=["POST"])
def generate_topics():
    user_input = request.json.get("user_input")
    if not user_input:
        return {"error": "user_input is required"}, 400

    try:
        topics = llm.generate_topics(llm, user_input)
        return {"topics": topics}, 200
    except Exception as e:
        return {"error": str(e)}, 500