import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(context_docs, user_query):
    context = "\n".join(context_docs)
    prompt = f"""
You are a helpful assistant for loan prediction Q&A.
Use the following context to answer:

Context:
{context}

User Question: {user_query}

Answer:
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You answer questions based on loan approval data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )
    return response["choices"][0]["message"]["content"].strip()

