# lab/llm/cohere_chain.py
import os
import json
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def cohere_model():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not set")
    return ChatCohere(model="command-a-03-2025", temperature=0.2, cohere_api_key=api_key)


def build_chain():
    prompt = PromptTemplate(input_variables=["prompt_text"], template="{prompt_text}")
    llm = cohere_model()
    parser = StrOutputParser()
    return prompt | llm | parser

def run_llm(prompt_text: str) -> dict:
    chain = build_chain()
    raw = chain.invoke({"prompt_text": prompt_text}).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("LLM did not return JSON")
    json_str = raw[start:end+1]
    return json.loads(json_str)
