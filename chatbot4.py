import random
import json
import pickle
import numpy as np
import nltk
import spacy
from keras.models import load_model
import os

path = os.getcwd()

#Se cargan los archivos que se creron anteriormente
intents_path = os.path.join(path, 'intents.json')
intents = json.loads(open(intents_path).read())

words_path = os.path.join(path, 'words.pkl')
words = pickle.load(open(words_path, 'rb'))

classes_path = os.path.join(path,'classes.pkl')
classes = pickle.load(open(classes_path, 'rb'))

model_path = os.path.join(path, 'chatbot_model.h5')
model = load_model(model_path)

nlp = spacy.load('es_core_news_sm')

#%%
def clean_up_sentences(sentence):
    doc = nlp(sentence)
    sentence_words = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

    return sentence_words

def bagw(sentence):
	#Se separan palabras de la frase de entrada
	sentence_words = clean_up_sentences(sentence)
	bag = [0]*len(words)
	for w in sentence_words:
		for i, word in enumerate(words):
            #Se comprueba si la palabra también está presente en la entrada
			if word == w:
				bag[i] = 1

	return np.array(bag)

#función que predecirá la clase de la oración ingresada por el usuario
def predict_class(sentence):
	bow = bagw(sentence)
	res = model.predict(np.array([bow]))[0]
	ERROR_THRESHOLD = 0.25
	results = [[i, r] for i, r in enumerate(res)
			if r > ERROR_THRESHOLD]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({'intent': classes[r[0]],
							'probability': str(r[1])})
		return return_list

#esta función imprimirá una respuesta aleatoria de cualquier clase a la que pertenezca la oración/palabras ingresadas por el usuario
def get_response(intents_list, intents_json):
	tag = intents_list[0]['intent']
	list_of_intents = intents_json['intents']
	result = ""
	for i in list_of_intents:
		if i['tag'] == tag:
			result = random.choice(i['responses'])
			break
	return result

print("!Chatbot está listo!")

#Se inicializa el chatbot
while True:
    message = input("")

    if message.lower() == "adiós":
        print("¡Adiós! Cuida tu salud y siempre consulta con especialistas.")
        break

    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)

