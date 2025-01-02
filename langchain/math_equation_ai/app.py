import base64
from io import BytesIO

import streamlit as st
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


def reset_chat():
    """
    Resets the chat history, chat agent, MathML content, and uploaded image.
    """
    st.session_state.chat_history = []
    st.session_state.chat_agent = None  # Reset the chat agent to start fresh
    st.session_state.mathml_content = None  # Reset MathML to allow re-upload
    global uploaded_image
    uploaded_image = None


def process_image_with_openai(image_file):
    """
    Processes the uploaded image to extract MathML content using OpenAI's vision model.

    Args:
        image_file: The uploaded image containing a mathematical expression.

    Returns:
        str: The extracted MathML content as plain text.
    """
    # Convert image to a byte buffer
    buffered = BytesIO()
    image_file.thumbnail((256, 256))  # Resize image to 256x256 for better processing
    image_file.save(buffered, format="PNG")  # Save image as PNG

    # Convert the image buffer to a base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Set up the OpenAI model for processing the image
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.5,
        max_tokens=1024,
    )

    # Send image and request MathML extraction
    response = model.invoke([
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Extract the MathML content from the provided image and return it as plain text. "
                        "Do not add any formatting, comments, or backticks around the response. "
                        "Only return the raw MathML content."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_str}"  # Embed the image as a base64 URL
                    },
                },
            ]
        )
    ])

    # Return the raw MathML content
    return response.content.strip()


def create_chat_chain():
    """
    Creates a LangChain chat agent using the extracted MathML content.

    Args:
        mathml_content: The extracted MathML content to serve as context for the chat.

    Returns:
        RunnableWithMessageHistory: The LangChain conversation agent initialized with the MathML content.
    """
    # Define the chat model
    chat_model = ChatOpenAI(model="gpt-4o", temperature=0.5)

    # Define a simple prompt template using MathML content
    prompt_template = """
        You are a helpful assistant with knowledge of mathematics. Here is some MathML content:
        {mathml_content}
        Please answer the following question:
        {question}
        """

    # Initialize the PromptTemplate with the MathML content
    prompt = PromptTemplate(
        input_variables=["mathml_content", "question"],
        template=prompt_template
    )
    # Chain the prompt with the chat model
    chain = prompt | chat_model

    return chain


# Streamlit Interface Setup
col1, col2 = st.columns([6, 1])
with col1:
    st.title("MathML Generator from Image with Chat using OpenAI")

with col2:
    st.button("Clear ↺", on_click=reset_chat)  # Button to clear chat history

# Initialize session state for MathML and chat history
if 'mathml_content' not in st.session_state:
    st.session_state.mathml_content = None
    st.session_state.chat_agent = None
    st.session_state.chat_history = []

# Handle file upload
uploaded_image = st.file_uploader("Upload an image with a math expression:", type=["jpg", "jpeg", "png"])
if uploaded_image:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Math Expression", use_container_width=True)

    if st.session_state.mathml_content is None:
        # Open the image for processing
        image = Image.open(uploaded_image)

        with st.spinner("Extracting math expression from image using OpenAI..."):
            st.session_state.mathml_content = process_image_with_openai(image)

    # Display the generated MathML content
    st.write("### Generated MathML")
    st.code(st.session_state.mathml_content, language="html")

    # Render the MathML content within HTML
    st.write("### Rendered Math Expression")
    mathml_code = f"""
    <html>
    <body>
        {st.session_state.mathml_content.strip()}
    </body>
    </html>
    """
    st.markdown(mathml_code, unsafe_allow_html=True)

    # Chat Interface
    st.write("### Chat About the MathML Content")

    # Initialize the chat agent with the extracted MathML content if not already initialized
    if st.session_state.chat_agent is None:
        st.session_state.chat_agent = create_chat_chain()

    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            st.chat_message(message["role"]).markdown(message["content"])

    # Chat input for user questions
    user_question = st.chat_input("Ask a question about the MathML content:")

    if user_question:
        # Append user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.chat_message("user").markdown(user_question)

        with st.spinner("Generating response..."):
            # Run the question through the chat agent to get an answer
            response = st.session_state.chat_agent.invoke({
                "mathml_content": st.session_state.mathml_content,
                "question": user_question
            })

            content = response.content

            # Append assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": content})
            st.chat_message("assistant").markdown(content)
