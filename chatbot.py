import random
import numpy as np
import json
import pickle

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer() # This is used to get the root word of a word, for example, the root word of "running" is "run"

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb')) # This loads the words list from the words.pkl file
classes = pickle.load(open('classes.pkl', 'rb')) # This loads the classes list from the classes.pkl file
model = load_model('chatbot_model.h5') # This loads the model from the chatbot_model.h5 file

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence) 
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words] # This loops through the sentence words and lemmatizes the words, and then adds them to the sentence words list
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words) # This creates a list of 0s that is the same length as the words list
    for s in sentence_words:
        for i, word in enumerate(words): # This loops through the words list and gets the index and the word
            if word == s:
                bag[i] = 1 # This sets the index of the bag list to 1 if the word is in the sentence words list
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0] # This gets the prediction from the model
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] # This loops through the results and only adds the results that are greater than the error threshold

    results.sort(key=lambda x: x[1], reverse=True) # This sorts the results by the highest probability
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent'] # tag is the intent with the highest probability
    list_of_intents = intents_json['intents'] # list of intents is the intents list from the intents.json file
    for i in list_of_intents:
        if i['tag'] == tag: #if the tag from the intents list matches the tag from the intents.json file
            result = random.choice(i['responses']) # make result equal to a random response from the responses list
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg)
    res = get_response(ints, intents)
    return res

print("Start talking with the bot (type quit to stop)!")

while True:
    message = input("")
    if message.lower() == "quit":
        break
    print(chatbot_response(message))