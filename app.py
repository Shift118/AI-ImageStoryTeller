from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())
#Img2Text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text
#llm
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
scenario = img2text("photo.jpg")
story = generate_story(scenario)

#Text to Speech