import gradio as gr
import litellm

# model_name = 'gemma3:1b'

messages_litellm = [
    {
        "role": "system", 
        "content": """
        You are a helpful assistant that provides clear and crisp answers to given questions about Science.
        You prioritise brevity and try to limit your answers to under 100 words."""
    }    
]

def inference(message, history, model_name):
    try:
        flattened_history = [item for sublist in (history or []) for item in sublist]
        full_message = " ".join(flattened_history + [message])
        messages_litellm.append({"role": "user", "content": full_message})
        partial_message = ""
        for chunk in litellm.completion(model=f"ollama/{model_name}",
                                        api_base="http://192.168.1.44:11434",
                                        messages=messages_litellm,
                                        max_new_tokens=512,
                                        temperature=.7,
                                        top_k=100,
                                        top_p=.9,
                                        repetition_penalty=1.18,
                                        stream=True):
            delta = chunk.get('choices', [{}])[0].get('delta', {})
            content = delta.get('content', '')  # Safely get content
            partial_message += content if content is not None else ''
            yield partial_message
        messages_litellm.append({"role": "assistant", "content": partial_message})
    except Exception as e:
        print("Exception encountered:", str(e))
        yield "An Error occurred please 'Clear' the error and try your question again"

with gr.Blocks() as demo:

    gr.Markdown("# Gradio + LiteLLM Chatbot")

    gr.ChatInterface(
        inference,
        type="messages",
        chatbot=gr.Chatbot(height=400, type='messages'),
        textbox=gr.Textbox(placeholder=f"Ask me a questiM4c_on ...", container=False, scale=5),
        theme="gradio/monochrome",
        additional_inputs=[
            gr.Dropdown(
            ["gemma3:1b", "deepseek-r1:1.5b"],
            label="Select a Model",
            info="PS: These are running locally."
            )
        ],
    )

demo.queue().launch()
