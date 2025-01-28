import warnings
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import requests
import os
import streamlit as st

# Suppress the specific warning about the slow image processor
warnings.filterwarnings("ignore", message="Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model.")

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Img2Text
def img2text(url):
    image_to_text = pipeline(
        "image-to-text", 
        model="Salesforce/blip-image-captioning-base", 
        use_fast=True  # Explicitly set use_fast=True
    )
    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text

# LLM
def generate_story(scenario):
    template = """
    you are a story teller;
    you can generate a short story based on a single narrative, the story should be no more than 200 words;
    
    CONTEXT: {scenario}
    STORY:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the Ollama model
    model = OllamaLLM(model="llama3.2")

    # Create a chain of the prompt and model
    chain = prompt | model
    
    # Generate the story using the provided scenario
    story = chain.invoke({"scenario": scenario})
    
    # Print the generated story
    print(story)
    
    return story

# Text to Speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"  # Updated model
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    payloads = {
        "inputs": message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    
    # Check if the response is successful
    if response.status_code == 200:
        with open('audio.mp3', 'wb') as file:
            file.write(response.content)
        print("Audio file saved successfully.")
    else:
        print(f"Error: {response.status_code}, {response.text}")

# Streamlit App
def main():
    st.title("Image to Story and Audio Generator")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded image temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Generate story from the image
        scenario = img2text("temp_image.jpg")
        st.write("**Generated Scenario:**")
        st.write(scenario)

        # Generate story using the scenario
        story = generate_story(scenario)
        st.write("**Generated Story:**")
        st.write(story)

        # Generate audio from the story
        text2speech(story)

        # Display the audio player
        st.write("**Generated Audio:**")
        st.audio("audio.mp3")

# Run the Streamlit app
if __name__ == "__main__":
    main()