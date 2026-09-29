import requests

URL = "http://127.0.0.1:8000/chat"

queries = [
    "What is Agentic AI?",
    "What are the key components of Agentic AI?",
    "What are the types and categories of agents?",
    "What is the role of planning in agentic systems?",
    "What are some applications of Agentic AI?"
]

for i, query in enumerate(queries, 1):
    response = requests.post(URL, json={"query": query})

    print(f"\nTEST {i}")
    print("Question:", query)
    print("Status:", response.status_code)
    print("Answer:", response.json()["final_answer"])
    print("Confidence:", response.json()["confidence_score"])

# Out-of-scope test
query = "What is the capital of France?"

response = requests.post(URL, json={"query": query})

print("\nOUT-OF-SCOPE TEST")
print("Question:", query)
print("Status:", response.status_code)
print("Answer:", response.json()["final_answer"])
print("Confidence:", response.json()["confidence_score"])