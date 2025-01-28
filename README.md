📸✨ **Img2Story2Audio** ✨🎙️  
**Turn Images into Stories and Audio with AI!**  

---

## 🚀 **What is this?**  
Welcome to **Img2Story2Audio** – the ultimate AI-powered tool for creative minds! 🧠💡  
This project takes an image, extracts its story, generates a creative narrative, and converts it into audio. Perfect for storytelling, creative writing, or just having fun with AI! 🎨📖🎧  

---

## 🌟 **Features**  
- **Image-to-Text** 🖼️➡️📝: Extract text descriptions from images using cutting-edge AI.  
- **Story Generation** 🧙‍♂️📖: Generate creative, short stories based on the image's context.  
- **Text-to-Speech** 🎙️🔊: Convert the generated story into lifelike audio.  
- **Streamlit UI** 🖥️: A sleek, user-friendly interface for seamless interaction.  

---

## 🛠️ **How It Works**  
1. **Upload an Image** 📤: Choose any image (JPG, PNG, etc.).  
2. **Extract the Scenario** 🔍: The AI analyzes the image and generates a text description.  
3. **Generate a Story** 🖋️: A creative story is crafted based on the image's context.  
4. **Listen to the Story** 🎧: The story is converted into audio, ready to play!  

---

## 🧑‍💻 **Tech Stack**  
- **AI Models**:  
  - 🖼️ **Image-to-Text**: `Salesforce/blip-image-captioning-base`  
  - 📖 **Story Generation**: `OllamaLLM` with `llama3.2`  
  - 🎙️ **Text-to-Speech**: Hugging Face's `espnet/kan-bayashi_ljspeech_vits`  
- **Frameworks**:  
  - 🐍 **Python**  
  - 🚀 **Streamlit** (for the UI)  
  - 🤗 **Hugging Face Transformers**  
  - 🔗 **LangChain** (for LLM integration)  

---

## 🚀 **Getting Started**  
### **Prerequisites**  
- Python 3.8+  
- A Hugging Face API token (for text-to-speech)  

### **Installation**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/Img2Story2Audio.git
   cd Img2Story2Audio
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your `.env` file:  
   - Add your Hugging Face API token:  
     ```plaintext
     HUGGINGFACE_API_TOKEN=your_token_here
     ```

4. Run the app:  
   ```bash
   streamlit run app.py
   ```

---

## 🎥 **Demo**  
![Demo GIF](https://media.giphy.com/media/your-demo-gif-url.gif)  
*Watch the magic happen!* ✨  

---

## 🤝 **Contributing**  
Love this project? Want to make it even better? Contributions are welcome! 🎉  
1. Fork the repository.  
2. Create a new branch:  
   ```bash
   git checkout -b feature/YourFeatureName
   ```  
3. Commit your changes:  
   ```bash
   git commit -m "Add some amazing feature"
   ```  
4. Push to the branch:  
   ```bash
   git push origin feature/YourFeatureName
   ```  
5. Open a pull request.  

---

## 📜 **License**  
This project is licensed under the MIT License. See the [LICENSE] file for details.  

---

## 🙏 **Acknowledgments**  
- **Hugging Face** for their amazing models and APIs.  
- **Ollama** for the powerful LLM integration.  
- **Streamlit** for making UI development a breeze.  

---

## 🚨 **Disclaimer**  
This project is for educational and fun purposes only. Use responsibly! 😄  