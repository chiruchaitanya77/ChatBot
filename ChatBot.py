import numpy as np
import tensorflow as tf
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Initialize the stemmer
stemmer = LancasterStemmer()

# Load intents file
with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

# Process intents to create training data
for intent in data["intents"]:
    if "text" in intent:
        for pattern in intent["text"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["intent"])

        if intent["intent"] not in labels:
            labels.append(intent["intent"])

# Stem and sort words
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

# Sort labels
labels = sorted(labels)

training = []
output = []

# Create output rows
out_empty = [0 for _ in range(len(labels))]

# Create training data
for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# Convert training data to numpy arrays
training = np.array(training)
output = np.array(output)

# Build the model using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(len(training[0]),)),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(training, output, epochs=1000, batch_size=8, verbose=1)

# Chatbot Interaction
print("Chatbot is ready! Type 'end' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "end":
        break

    # Preprocess user input
    user_words = nltk.word_tokenize(user_input)
    user_bag = [0] * len(words)
    user_bag = [1 if stemmer.stem(w.lower()) in user_words else 0 for w in words]

    # Get model prediction
    prediction = model.predict(np.array([user_bag]))
    intent_index = np.argmax(prediction)
    tag = labels[intent_index]

    # Find the corresponding intent and response
    for intent in data["intents"]:
        if intent["intent"] == tag:
            response = random.choice(intent["responses"])
            break

    print("Chatbot: " + response)
