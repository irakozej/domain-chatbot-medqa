import gradio as gr
from transformers import T5TokenizerFast, TFAutoModelForSeq2SeqLM
import tensorflow as tf

# ==============================
# 🧠 Load model and tokenizer
# ==============================
MODEL_PATH = "/Users/mac/Schools/ALU/ML Techniques I/domain-chatbot-medqa/models/t5_medqa_finetuned"
PREFIX = "question: "
MAX_LEN = 128

print("🔄 Loading model and tokenizer...")
tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
print("✅ Model loaded successfully!")

# ==============================
# 💬 Chatbot function
# ==============================
def chatbot_response(user_message, history):
    if not user_message.strip():
        return history

    # Format input
    input_text = PREFIX + user_message
    inputs = tokenizer.encode(input_text, return_tensors="tf", max_length=MAX_LEN, truncation=True)

    # Generate response
    output = model.generate(
        inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True,
        temperature=0.7,
    )
    bot_reply = tokenizer.decode(output[0], skip_special_tokens=True)

    # Append to chat history
    history.append((user_message, bot_reply))
    return history

# ==============================
# 🎨 Custom CSS for styling
# ==============================
custom_css = """
.chatbot-container {
    background-color: #f9fafb;
    border-radius: 15px;
    padding: 20px;
}

.user-bubble {
    background-color: #2563eb;
    color: white;
    border-radius: 20px;
    padding: 10px 15px;
    max-width: 75%;
    align-self: flex-end;
    margin: 5px;
}

.bot-bubble {
    background-color: #e5e7eb;
    color: #111827;
    border-radius: 20px;
    padding: 10px 15px;
    max-width: 75%;
    align-self: flex-start;
    margin: 5px;
}

footer {
    display: none !important;
}
"""

# ==============================
# ⚙️ Gradio UI
# ==============================
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style='text-align:center;'>
            <h1 style='color:#2563eb;'>💬 MedQA Chatbot</h1>
            <p style='color:gray;'>Your AI assistant for medical questions</p>
        </div>
        """,
    )

    chatbot = gr.Chatbot(label="MedQA Chatbot", elem_classes=["chatbot-container"])
    user_input = gr.Textbox(
        placeholder="Ask me anything medical...",
        label="Your Question",
        lines=1
    )

    with gr.Row():
        clear_btn = gr.Button("🧹 Clear Chat", variant="secondary")
        send_btn = gr.Button("🚀 Send", variant="primary")

    # Function linking
    send_btn.click(chatbot_response, [user_input, chatbot], [chatbot])
    user_input.submit(chatbot_response, [user_input, chatbot], [chatbot])
    clear_btn.click(lambda: None, None, chatbot, queue=False)

# ==============================
# 🚀 Launch app
# ==============================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
