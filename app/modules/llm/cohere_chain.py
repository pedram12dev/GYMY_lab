import os, json
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def _cohere():
    key = os.getenv("COHERE_API_KEY")
    if not key: raise RuntimeError("COHERE_API_KEY not set")
    return ChatCohere(model="command-a-03-2025", temperature=0.2, cohere_api_key=key)

def _chain():
    prompt = PromptTemplate(input_variables=["prompt_text"], template="{prompt_text}")
    return prompt | _cohere() | StrOutputParser()

def run_llm(prompt_text: str) -> dict:
    raw = _chain().invoke({"prompt_text": prompt_text}).strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1: raise ValueError("LLM did not return JSON")
    return json.loads(raw[s:e+1])
