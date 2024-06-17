import random
import numpy as np
import json
import nltk
import spacy
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pickle

#nltk.download('omw-1.4')
#nltk.download("punkt")
#nltk.download("wordnet")

import sys
print(sys.version)

spacy.cli.download('es_core_news_sm')
nlp = spacy.load('es_core_news_sm')

#Carga de datos
def load_intents(path):
    with open(path, 'r') as file:
        intents = json.load(file)
    return intents

#Preprocesamiento
def preprocess_data(intents):
    words = []
    classes = []
    documents = []

    ignore_words = ['?', '!']

    for intent in intents['intents']:
        #Itera a través de cada patrón asociado con una intención
        for pattern in intent['patterns']:
            #Tokenización con SpaCy (español)
            tokens = [token.lemma_ for token in nlp(pattern) if token.text not in ignore_words]
            #Añade las palabras lematizadas a la lista de palabras
            words.extend(tokens)
            #Añade una tupla de (tokens, tag de la intención) a la lista de documentos
            documents.append((tokens, intent['tag']))
              #Si el tag de la intención no está ya en la lista de clases, añadirlo
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
     #Ordena y elimina duplicados para obtener una lista final de palabras únicas
    words = sorted(list(set(words)))
     #Ordena y elimina duplicados para obtener una lista final de clases únicas
    classes = sorted(list(set(classes)))

    #Devuelve las palabras, clases y documentos procesados
    return words, classes, documents

#Creación de datos de entrenamiento
def create_training_data(words, classes, documents):
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        #Se crea una 'bolsa de palabras' para el documento actual
        bag = [1 if word in doc[0] else 0 for word in words]
        #Se copia la lista de salida vacía para preparar la fila de salida para este documento
        output_row = list(output_empty)
        #Se marca con 1 la posición correspondiente a la clase del documento actual en la fila de salida
        output_row[classes.index(doc[1])] = 1
        #Se añade la bolsa de palabras y la fila de salida correspondiente a la lista de entrenamiento
        training.append([bag, output_row])

    #Se mezcla aleatoriamente los datos de entrenamiento para asegurar variedad durante el entrenamiento
    random.shuffle(training)
    training = np.array(training, dtype=object)

    #Se separan los datos en características (train_x) e etiquetas (train_y)
    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    # Devolver las características y etiquetas preparadas para el entrenamiento
    return train_x, train_y

#%%%
#Construcción del modelo de red neuronal
#OPCION 1: CON ACCURACY DE ~70%
def build_model_updated(input_shape, output_shape):
    model = Sequential([
        LSTM(128, input_shape=(input_shape, 1)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(output_shape, activation='softmax')
    ])
    optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
#%%%
#Construcción del modelo de red neuronal
#OPCION 2: CON ACCURACY DE ~80%
#VER BIEN LO DE ADAM

from keras.optimizers import Adam

def build_model_updated(input_shape, output_shape):
    model = Sequential([
        LSTM(128, input_shape=(input_shape, 1), return_sequences=True),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_shape, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

#%%%
intents = load_intents("intents.json")
words, classes, documents = preprocess_data(intents)

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

train_x, train_y = create_training_data(words, classes, documents)

#Se inicilaiza modelo
model = build_model_updated(len(train_x[0]), len(train_y[0]))

#Se entrena
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save("chatbot_model.h5")

model.save("Chatbot_model2.keras")
#%%

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

