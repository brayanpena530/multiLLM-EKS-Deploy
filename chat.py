from openai import OpenAI
import gradio as gr
import requests

# Define model-to-URL mapping
model_to_url = {
    "TinyLlama": "http://localhost:8000/v1",
    "Jina Embeddings": "http://localhost:8080/",
    "Conan Embeddings": "http://localhost:8081/"
}

# Define the predict function
def predict(message, history, temperature, max_tokens, selected_model):

    # Get the base URL based on the selected model
    base_url = model_to_url[selected_model]

    # Create a client
    client = OpenAI(base_url=base_url, api_key="test-key")

    # If selected model is TinyLlama, treat it as a chat model
    if selected_model == "TinyLlama":
        history_openai_format = []
        
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human})
            history_openai_format.append(
                {"role": "assistant", "content": assistant})

        history_openai_format.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            messages=history_openai_format,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                partial_message = partial_message + chunk.choices[0].delta.content
                yield partial_message

    # If selected model is an embedding model, handle it differently
    else:
        response = requests.post(
            f"{base_url}embed",
            json={"inputs":message}
        )
        response_data = response.json()
        yield str(response_data)


if __name__ == "__main__":
    # Dropdown for model selection
    model_dropdown = gr.Dropdown(
        label="Select Model",
        choices=list(model_to_url.keys()),
        value="TinyLlama",
        interactive=True
    )

    # Launch the chat interface
    gr.ChatInterface(
        fn=predict,
        additional_inputs=[
            # gr.Textbox(label="User Input"),
            # gr.State([]),  # For history
            gr.Slider(
                label="Temperature",
                value=0.5,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                interactive=True,
                info="Higher values produce more diverse outputs"
            ),
            gr.Slider(
                label="Max new tokens",
                value=500,
                minimum=0,
                maximum=2048,
                step=64,
                interactive=True,
                info="The maximum numbers of new tokens"
            ),
            model_dropdown  # Adding model selection as input
        ],
        title="GenAI Platform Demo",
        description="GenAI Platform Chatbot",
        additional_inputs_accordion=gr.Accordion(open=True, label="Settings")
    ).launch()
