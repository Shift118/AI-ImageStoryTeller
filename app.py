from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import requests
import os
import warnings

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Suppress the specific warning about the slow image processor
warnings.filterwarnings("ignore", message="Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model.")


# Img2Text
def img2text(url):
    image_to_text = pipeline(
        "image-to-text", 
        model="Salesforce/blip-image-captioning-base", 
        use_fast=True  # Suppress the warning
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
    model = OllamaLLM(model="llama3.2:1b")

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

# Main workflow
scenario = img2text("photo.jpg")  # Replace with your image path
story = generate_story(scenario)
text2speech(story)