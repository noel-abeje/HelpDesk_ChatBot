import random
import numpy as np
import json # json is a module that is used to load json files into python
import pickle # pickle is a module that is used to serialize and deserialize python objects into a byte stream so that they can be saved to a file

import nltk
from nltk.stem import WordNetLemmatizer # This is used to get the root word of a word, for example, the root word of "running" is "run"

from tensorflow.keras.models import Sequential # This is used to create a neural network, which is a series of layers that are connected to each other and are used to train the model
from tensorflow.keras.layers import Dense, Activation, Dropout # Dense is used to create a fully connected layer, Activation is used to create an activation function, Dropout is used to prevent overfitting
from tensorflow.keras.optimizers import legacy # SGD is used to create a Stochastic Gradient Descent optimizer, which is used to train the model

lemmatizer = WordNetLemmatizer() # This is used to get the root word of a word, for example, the root word of "running" is "run"

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ','] # These are the characters that we want to ignore because they don't add any value to the model

for intent in intents['intents']: # This loops through the intents in the intents.json file
    for pattern in intent['patterns']: # This loops through the patterns in the intents.json file
        word_list = nltk.word_tokenize(pattern) # What tokenization does is that it splits the sentence into words, for example, "Hello, how are you?" will be split into "Hello", ",", "how", "are", "you", "?"
        words.extend(word_list) # This adds the word list to the words list
        documents.append((word_list, intent['tag'])) # This adds the word list and the tag to the documents list

        if intent['tag'] not in classes:
            classes.append(intent['tag']) # This adds the tag to the classes list

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters] # This loops through the words list and lemmatizes the words, and then adds them to the words list
words = sorted(set(words)) # This sorts the words list and removes any duplicates

classes = sorted(set(classes)) # This sorts the classes list and removes any duplicates

pickle.dump(words, open('words.pkl', 'wb')) # This creates a pickle file that contains the words list and saves it as words.pkl as a binary file
pickle.dump(classes, open('classes.pkl', 'wb')) # This creates a pickle file that contains the classes list and saves it as classes.pkl as a binary file

#Neural Networks cant work with strings, so we need to convert the words and classes into numbers so that the neural network can understand them
#Neural Networks work by creating a series of layers that are connected to each other, and each layer has a set of weights that are used to train the model

training = []
output_empty = [0] * len(classes) # This creates a list of 0s that is the same length as the classes list

for document in documents: # This loops through the documents list
    bag = []
    word_patterns = document[0] # This gets the word list from the documents list
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns] # This loops through the word list and lemmatizes the words, and then adds them to the bag list

    for word in words: # This loops through the words list
        bag.append(1) if word in word_patterns else bag.append(0) # This adds a 1 to the bag list if the word is in the word list, and adds a 0 if it isn't

    output_row = list(output_empty) # This creates a copy of the output_empty list
    output_row[classes.index(document[1])] = 1 # This sets the index of the output_row list to 1 if the tag is in the classes list

    training.append([bag, output_row]) # This adds the bag list and the output_row list to the training list

random.shuffle(training) # This shuffles the training list
training = np.array(training) # This converts the training list into a numpy array

train_x = list(training[:, 0]) # This gets the first column of the training array and adds it to the train_x list
train_y = list(training[:, 1]) # This gets the second column of the training array and adds it to the train_y list

model = Sequential() # This creates a sequential model
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # This creates a fully connected layer with 128 neurons, the input shape is the length of the train_x list, and the activation function is relu
model.add(Dropout(0.5)) # This creates a dropout layer with a rate of 0.5, which means that 50% of the neurons will be randomly disabled during training
model.add(Dense(64, activation='relu')) # This creates a fully connected layer with 64 neurons, and the activation function is relu
model.add(Dropout(0.5)) # This creates a dropout layer with a rate of 0.5, which means that 50% of the neurons will be randomly disabled during training
model.add(Dense(len(train_y[0]), activation='softmax')) # This creates a fully connected layer with the same number of neurons as the train_y list, and the activation function is softmax. They all have to add up to 1

sgd = legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # This creates a Stochastic Gradient Descent optimizer with a learning rate of 0.01, a decay of 1e-6, a momentum of 0.9, and nesterov set to True. Nesterov is used to accelerate the gradient descent
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # This compiles the model with a loss function of categorical_crossentropy, the optimizer is the sgd optimizer, and the metrics is accuracy

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) # This trains the model with the train_x list, the train_y list, 200 epochs, a batch size of 5, and verbose set to 1 so that it shows the progress of the training process
model.save('chatbot_model.h5', hist) # This saves the model as chatbot_model.h5

print("Done")
