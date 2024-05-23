import gradio as gr
import requests

def generate_response(prompt):
    response = requests.post("http://localhost:8000/generate", json={"prompt": prompt})
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error: " + response.json()["detail"]

iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="RAG Application",
    description="Enter a prompt to generate a response using LangChain, Qdrant, and OpenAI."
)

if __name__ == "__main__":
    iface.launch()
