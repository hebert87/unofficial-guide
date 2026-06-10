"""
Gradio query interface for The Unofficial Guide (Milestone 5).

Run:
    python app.py
Then open http://localhost:7860
"""

import gradio as gr

from query import ask


def handle_query(question):
    if not question or not question.strip():
        return "Please enter a question.", ""
    result = ask(question)
    if result["sources"]:
        sources = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources = "(no sources — the guide didn't have enough information)"
    return result["answer"], sources


with gr.Blocks(title="The Unofficial Guide") as demo:
    gr.Markdown(
        "# 🏠 The Unofficial Guide\n"
        "Ask about student housing near Green River College. Answers come only "
        "from real student posts and housing docs — with sources cited."
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. How much do students pay for rent near campus?",
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    gr.Examples(
        examples=[
            "How much do students pay for rent near campus?",
            "How do I make sure I get my security deposit back?",
            "How can I avoid rental scams when apartment hunting?",
            "What are my options for summer housing on a 12-month lease?",
        ],
        inputs=inp,
    )

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
