import numpy as np
import nltk
import re
import random

from nltk.tokenize import word_tokenize

# Download tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

# Training dataset
data = {
    "hello": [
        "Hello! How can I help you?",
        "Hi! Nice to meet you."
    ],
    "how are you": [
        "I am fine, thank you!",
        "I am doing great!"
    ],
    "what is your name": [
        "I am a simple Python chatbot."
    ],
    "what is python": [
        "Python is a popular programming language."
    ],
    "thank you": [
        "You're welcome!",
        "Happy to help!"
    ],
    "bye": [
        "Goodbye!",
        "Have a nice day!"
    ]
}

# Convert text into tokens
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return word_tokenize(text)

# Convert words into a NumPy vector
def vectorize(text, vocabulary):
    tokens = preprocess(text)
    vector = np.zeros(len(vocabulary))

    for word in tokens:
        if word in vocabulary:
            vector[vocabulary.index(word)] += 1

    return vector

# Create vocabulary
vocabulary = []

for sentence in data.keys():
    vocabulary.extend(preprocess(sentence))

vocabulary = sorted(set(vocabulary))

# Calculate similarity
def similarity(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0

    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

# Generate chatbot response
def chatbot(user_input):

    user_vector = vectorize(user_input, vocabulary)

    best_sentence = None
    best_score = 0

    for sentence in data.keys():

        sentence_vector = vectorize(sentence, vocabulary)

        score = similarity(user_vector, sentence_vector)

        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_score > 0:
        return random.choice(data[best_sentence])

    return "Sorry, I don't understand that."

# Chatbot interaction
print("Chatbot: Hello! Type 'bye' to exit.")

while True:

    user_input = input("You: ")

    response = chatbot(user_input)

    print("Chatbot:", response)

    if user_input.lower() == "bye":
        break