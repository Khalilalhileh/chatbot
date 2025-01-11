import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

url = 'twcs.csv'  
df = pd.read_csv(url, nrows=100000)

customer_queries = df[df['inbound'] == True]
company_responses = df[df['inbound'] == False]

customer_queries = customer_queries[['response_tweet_id', 'text']].dropna().reset_index(drop=True)
company_responses = company_responses[['tweet_id', 'text']].dropna().reset_index(drop=True)

customer_queries['response_tweet_id'] = customer_queries['response_tweet_id'].astype(str)
company_responses['tweet_id'] = company_responses['tweet_id'].astype(str)

conversation_data = pd.merge(
    customer_queries, 
    company_responses, 
    left_on='response_tweet_id', 
    right_on='tweet_id', 
    suffixes=('_query', '_response')
)

if len(conversation_data) >= 100000:
    conversation_data = conversation_data.sample(n=100000, random_state=42).reset_index(drop=True)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(
    conversation_data['text_query'].tolist() + 
    conversation_data['text_response'].tolist()
)

query_sequences = tokenizer.texts_to_sequences(conversation_data['text_query'].tolist())
response_sequences = tokenizer.texts_to_sequences(conversation_data['text_response'].tolist())

MAX_SEQUENCE_LENGTH = 50
query_sequences_padded = pad_sequences(query_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
response_sequences_padded = pad_sequences(response_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

decoder_input_data = response_sequences_padded[:, :-1]  
decoder_output_data = np.expand_dims(response_sequences_padded[:, 1:], -1)  

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 64
lstm_units = 128

query_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
decoder_input = Input(shape=(MAX_SEQUENCE_LENGTH - 1,))

embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

query_embedded = embedding_layer(query_input)
decoder_embedded = embedding_layer(decoder_input)

query_lstm = LSTM(lstm_units, return_sequences=True)(query_embedded)
decoder_lstm = LSTM(lstm_units, return_sequences=True)(decoder_embedded)

decoder_output = Dense(vocab_size, activation='softmax')(decoder_lstm)

model = Model(inputs=[query_input, decoder_input], outputs=decoder_output)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

checkpoint = ModelCheckpoint(
    'customer_support_chatbot_model.h5', 
    monitor='val_loss', 
    save_best_only=True, 
    verbose=1
)

train_size = int(0.8 * len(query_sequences_padded))
X_train = [query_sequences_padded[:train_size], decoder_input_data[:train_size]]
y_train = decoder_output_data[:train_size]

X_val = [query_sequences_padded[train_size:], decoder_input_data[train_size:]]
y_val = decoder_output_data[train_size:]

EPOCHS = 5
BATCH_SIZE = 64

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint]
)

import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Model training completed and saved as 'customer_support_chatbot_model.h5'.")
print("Tokenizer saved as 'tokenizer.pkl'.")