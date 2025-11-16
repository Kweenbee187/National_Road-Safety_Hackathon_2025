import gradio as gr
from rag_engine import run_query

TITLE = "Road Safety Intervention GPT 🚧"
DESC = """
AI system that recommends **correct IRC-based road safety interventions**  
using RAG + Groq Llama 3.1 + FAISS.  
Team MUFFIN — Sneha Chakraborty & Divyansh Pathak
"""

def process_issue(issue):
    if not issue.strip():
        return "Please enter a road safety issue."
    return run_query(issue)

demo = gr.Interface(
    fn=process_issue,
    inputs=gr.Textbox(lines=4, label="Describe the road safety issue"),
    outputs=gr.Markdown(label="Recommendation"),
    title=TITLE,
    description=DESC,
)

demo.launch()
