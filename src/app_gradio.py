# src/app_gradio.py
import gradio as gr
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# =========================================================
# ✅ Load Model and Tokenizer
# =========================================================
MODEL_PATH = "/Users/mac/Schools/ALU/ML Techniques I/domain-chatbot-medqa/models/t5_medqa_finetuned"

print(f"🔍 Loading TensorFlow model from local path: {MODEL_PATH}")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = TFT5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# =========================================================
# ✅ Chat History Management
# =========================================================
chat_history = []  # list of tuples: (user, bot)

def medqa_chat(user_input, history):
    if not user_input.strip():
        return history, gr.update(value="")

    input_text = "question: " + user_input
    input_ids = tokenizer.encode(input_text, return_tensors="tf")

    output = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    bot_reply = tokenizer.decode(output[0], skip_special_tokens=True)

    history = history + [(user_input, bot_reply)]
    return history, gr.update(value="")

def clear_chat():
    return [], gr.update(value="")

# =========================================================
# ✅ Custom Chat UI Styling
# =========================================================
css = """
#chatbot {
  height: 500px !important;
  overflow-y: auto !important;
}

.user-message {
  background-color: #DCF8C6;
  color: black;
  padding: 8px 12px;
  border-radius: 16px 16px 0 16px;
  margin: 6px 0;
  max-width: 75%;
  align-self: flex-end;
}

.bot-message {
  background-color: #EAEAEA;
  color: black;
  padding: 8px 12px;
  border-radius: 16px 16px 16px 0;
  margin: 6px 0;
  max-width: 75%;
  align-self: flex-start;
}

#chatbox {
  display: flex;
  flex-direction: column;
}
"""

# =========================================================
# ✅ Create Chatbot Interface
# =========================================================
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🩺 MedQA Chatbot — Ask Medical Questions")
    chatbot = gr.Chatbot(elem_id="chatbot", label="Chat History", height=500)
    user_input = gr.Textbox(
        placeholder="Type your medical question here...",
        show_label=False
    )
    clear_btn = gr.Button("🧹 Clear Chat")

    with gr.Row():
        submit_btn = gr.Button("💬 Send", variant="primary")

    # Events
    submit_btn.click(medqa_chat, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(medqa_chat, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    clear_btn.click(clear_chat, outputs=[chatbot, user_input])

# =========================================================
# ✅ Launch App
# =========================================================
if __name__ == "__main__":
    demo.launch(share=True)
