
# AI Intent-Based Chatbot with TensorFlow

A simple yet powerful Python chatbot built using **TensorFlow (Keras)** and **Natural Language Processing (NLP)** techniques. This bot is designed to understand user intents from a JSON dataset, train a neural network, and provide responses with an associated confidence score.

## 🚀 Features

* **Deep Learning Model:** Uses a Sequential Neural Network with Embedding and Global Average Pooling layers.
* **Intent Recognition:** Classifies user text into specific tags (intents) defined in a JSON file.
* **Confidence Scoring:** Displays how confident the model is in its prediction (0-100%).
* **Dynamic Response:** Randomly selects a response from a pool of answers associated with the matched intent.
* **Scalable:** Easily expandable by adding more patterns and tags to the `intents.json` file.

## 🛠️ Prerequisites

Ensure you have Python installed (3.7+ recommended). You will need to install the following libraries:

```bash
pip install tensorflow numpy scikit-learn pyttsx3

```

*Note: `pyttsx3` is imported for Text-to-Speech capabilities but is optional if you only want text output.*

## 📂 Project Structure

The project relies on two main files:

1. `chatbot.py` (The Python script provided).
2. `intents.json` (The database of conversation patterns).

### Setting up `intents.json`

**Crucial Step:** You must create a file named `intents.json` in the same directory as your script. This file tells the bot what to listen for and how to respond.

**Example `intents.json` structure:**

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey", "Good morning", "Is anyone there?"],
      "responses": ["Hello!", "Hi there!", "Greetings!", "How can I help you?"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye", "I'm leaving"],
      "responses": ["Goodbye!", "See you soon!", "Have a nice day!"]
    },
    {
      "tag": "thanks",
      "patterns": ["Thanks", "Thank you", "That's helpful"],
      "responses": ["You're welcome!", "Happy to help!", "My pleasure."]
    },
    {
      "tag": "name",
      "patterns": ["What is your name?", "Who are you?"],
      "responses": ["I am a chatbot created with TensorFlow.", "You can call me Bot."]
    }
  ]
}

```

## 🏃‍♂️ How to Run

1. Save your python script as `chatbot.py`.
2. Ensure `intents.json` is in the same folder.
3. Run the script via terminal or command prompt:

```bash
python chatbot.py

```

4. The model will train (you will see epoch progress bars).
5. Once trained, the bot will start:
```text
ChatBot: Hello! How can I assist you today?
User: Hello
ChatBot (greeting, Confidence: 99.85%): Hi there!

```



## 🧠 How it Works

1. **Data Loading:** The script loads the `intents.json` file.
2. **Preprocessing:**
* **Tokenization:** Converts words into numerical integers.
* **Padding:** Ensures all input sentences are the same length (`max_len`).
* **Label Encoding:** Converts text tags (e.g., "greeting") into numbers.


3. **Model Architecture:**
* **Embedding Layer:** Converts integer tokens into dense vectors of fixed size.
* **GlobalAveragePooling1D:** Averages the vector to handle variable-length inputs efficiently.
* **Dense Layers:** Fully connected layers to classify the input into the correct tag.


4. **Inference:** When you type a message, the model predicts the tag, calculates the confidence score, and retrieves a random response for that tag.

## ⚙️ Configuration

You can tweak the hyperparameters inside the script to improve performance:

* **`max_len`**: (Default `20`) Change this if you expect very long user inputs.
* **`epochs`**: (Default `100`) Increase this if the loss isn't decreasing enough during training.
* **`embedding_dim`**: (Currently `16`) Increase to `32` or `64` for more complex vocabulary.

## 🤝 Contributing

Feel free to fork this project and submit pull requests. You can extend this by adding:

* Text-to-Speech integration (using the imported `pyttsx3`).
* Saving/Loading the trained model (`model.save()`) so you don't have to retrain every time.
* A GUI using Tkinter or PyQt.

---

### Would you like me to help you create a robust `intents.json` file to get you started?

