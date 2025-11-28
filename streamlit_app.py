import streamlit as st
from llama_cpp import Llama

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses a local Llama model to generate responses. "
)

# Load the model using from_pretrained (cached in session state to avoid reloading)
@st.cache_resource
def load_model():
    """Load the Llama model from Hugging Face Hub."""
    print("Loading model from Hugging Face Hub...")
    llm = Llama.from_pretrained(
        repo_id="kinga-anna/lora_model_merged-Q4_K_M-GGUF",
        filename="lora_model_merged-q4_k_m.gguf",
        n_ctx=2048,  # Context window
        n_threads=4,  # Adjust based on your CPU
        n_gpu_layers=0,  # Set to higher value if you have GPU support
        verbose=False,
    )
    return llm


def format_prompt(message: str, history: list) -> str:
    """Format the conversation using Llama 3.1 chat template."""
    system_prompt = "Cutting Knowledge Date: December 2023\nToday Date: 26 July 2024\n\n"

    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    prompt += system_prompt
    prompt += "<|eot_id|>"

    # Add conversation history
    for user_msg, assistant_msg in history:
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}<|eot_id|>"

    # Add current message
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt


def chat(llm, message: str, history: list) -> str:
    """Generate a response from the model."""
    prompt = format_prompt(message, history)

    output = llm(
        prompt,
        max_tokens=512,
        temperature=1.5,
        min_p=0.1,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        echo=False,
    )

    response = output["choices"][0]["text"].strip()
    return response


# Load the model
llm = load_model()

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the local Llama model.
    # Convert message history to format expected by chat function
    history = []
    for i in range(0, len(st.session_state.messages) - 1, 2):
        if i + 1 < len(st.session_state.messages):
            history.append((
                st.session_state.messages[i]["content"],
                st.session_state.messages[i + 1]["content"]
            ))

    with st.chat_message("assistant"):
        # Get the current user message
        current_message = st.session_state.messages[-1]["content"]
        response = chat(llm, current_message, history)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
