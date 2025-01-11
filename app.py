from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import firebase_admin
from firebase_admin import credentials, db
app = Flask(__name__)

model = load_model('customer_support_chatbot_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_SEQUENCE_LENGTH = 50  
START_TOKEN = 1  
END_TOKEN = 2    
cred = credentials.Certificate("customer-support-chatbot-feb26-firebase-adminsdk-lgqmw-5c9ca00a64.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://customer-support-chatbot-feb26-default-rtdb.firebaseio.com/'
})
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'response': "Please provide a valid input."})

    query_sequence = tokenizer.texts_to_sequences([query])
    query_padded = pad_sequences(query_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    decoder_input_data = np.zeros((1, MAX_SEQUENCE_LENGTH - 1), dtype=np.int32)  
    decoder_input_data[0, 0] = START_TOKEN

    response_tokens = []
    for i in range(MAX_SEQUENCE_LENGTH - 1):  
        predictions = model.predict([query_padded, decoder_input_data])
        next_token_index = np.argmax(predictions[0, i, :])

        if next_token_index == END_TOKEN:
            break

        response_tokens.append(next_token_index)

        if i + 1 < MAX_SEQUENCE_LENGTH - 1:
            decoder_input_data[0, i + 1] = next_token_index

    response_words = [tokenizer.index_word.get(token, "") for token in response_tokens]
    response_text = " ".join(response_words).strip()

    if not response_text:
        response_text = "Sorry, I couldn't find an appropriate response."
    ref = db.reference('chatbot')
    ref.push({
        'query': query,
        'response': response_text
    })

    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)