import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""  # set OpenAI API key
client = OpenAI()  # reads API key from env

# openai.api_key

def generate_answer_with_context(query, context, max_tokens=200):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely:"

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7
    )

    return response.choices[0].message.content
