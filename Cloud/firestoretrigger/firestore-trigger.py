from google.cloud import firestore
import pytz
from datetime import datetime

# Codingan Chatbot
from flask import Flask, request, jsonify
import tensorflow
import numpy as np
import json
import re
from google.cloud import storage
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM


def hello_world(text_fire):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('ml_model415_indo')

    inputs = text_fire.lower()
    keyword = bool(re.search(r'fisika', inputs))

    if keyword:
        inputs = re.sub(r'fisika ','',inputs)
        ######### About Fisika ########
        ######### load ########
        answers_txt = bucket.blob('AnswerFisika.txt')
        questions_txt = bucket.blob('QuestionFisika.txt')
        encoder_weights = bucket.blob('enc_model_fisika.h5')
        decoder_weights = bucket.blob('dec_model_fisika.h5')
        model_weights = bucket.blob('main_model_fisika.h5')


    else:
        ######## About Robot ########
        ######## load ########
        answers_txt = bucket.blob('Answers.txt')
        questions_txt = bucket.blob('Questions.txt')
        encoder_weights = bucket.blob('enc_model.h5')
        decoder_weights = bucket.blob('dec_model.h5')
        model_weights = bucket.blob('main_model.h5')
        
    ######### download ########
    answers_txt.download_to_filename('/tmp/Answers.txt')
    questions_txt.download_to_filename('/tmp/Questions.txt')
    encoder_weights.download_to_filename('/tmp/enc_model.h5')
    decoder_weights.download_to_filename('/tmp/dec_model.h5')
    model_weights.download_to_filename('/tmp/main_model.h5')
    
    questions = open('/tmp/Questions.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    answers = open('/tmp/Answers.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

    def clean_text(text):
        text = text.lower().strip()
        text = re.sub(r'" - "+', "", text)
        text = re.sub(r'" +"', " ", text)
        text = re.sub(r'"^ "', "", text)
        text = re.sub(r'[-()\"#/@;:<>{}`+=~|.!?,]', "", text)
        text = re.sub(r'[\-\-]', "", text)
        text = re.sub(r'\.\.\.', "", text)
        text = re.sub(r'^- ', "", text)
        return text

    # Clean Questions
    clean_questions = []
    for question in questions:
        clean_questions.append(clean_text(question))
        
    # Clean Answers
    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_text(answer))

    # Dictionary that maps word to its number occurances 
    # To set the input of embedding layers
    word2count = {}
    for question in clean_questions:
        for word in question.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    for answer in clean_answers:
        for word in answer.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    print(len(word2count))

    ###  remove less frequent ###
    threshold = 5

    vocab = {}
    word_num = 0
    for word, count in word2count.items():
        if count >= threshold:
            vocab[word] = word_num
            word_num += 1

    for i in range(len(clean_answers)):
        clean_answers[i] = '<SOS> ' + clean_answers[i] + ' <EOS>'

    tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
    x = len(vocab)
    for token in tokens:
        vocab[token] = x
        x += 1
        
    inv_vocab = {w:v for v, w in vocab.items()}

    encoder_inp = []
    for line in clean_questions:
        lst = []
        for word in line.split():
            if word not in vocab:
                lst.append(vocab['<OUT>'])
            else:
                lst.append(vocab[word])
            
        encoder_inp.append(lst)
        
    decoder_inp = []
    for line in clean_answers:
        lst = []
        for word in line.split():
            if word not in vocab:
                lst.append(vocab['<OUT>'])
            else:
                lst.append(vocab[word])        
        decoder_inp.append(lst)

    max_input_len = 15
    lstm_layers = 400
    VOCAB_SIZE = len(vocab)

    encoder_inp = pad_sequences(encoder_inp, maxlen = max_input_len, padding='post', truncating='post')
    decoder_inp = pad_sequences(decoder_inp, maxlen = max_input_len, padding='post', truncating='post')

    decoder_final_output = []
    for i in decoder_inp:
        decoder_final_output.append(i[1:])

    decoder_final_output = pad_sequences(decoder_final_output, maxlen = max_input_len, padding='post', truncating='post')
    decoder_final_output = to_categorical(decoder_final_output, len(vocab))

    embedding = Embedding(VOCAB_SIZE + 1,
                        output_dim = 50, 
                        input_length = max_input_len,
                        trainable = True,
                        mask_zero = True
                        )

    ######################################### Encode #########################################
    e_input = Input(shape=(max_input_len, ))
    d_input = Input(shape=(max_input_len, ))

    e_embeded = embedding(e_input)
    e_lstm = LSTM(lstm_layers, return_sequences=True, return_state=True)
    e_op, h_state, c_state = e_lstm(e_embeded)
    e_states = [h_state, c_state]

    d_embeded = embedding(d_input)
    d_lstm = LSTM(lstm_layers, return_sequences=True, return_state=True)
    d_output, _, _ = d_lstm(d_embeded, initial_state = e_states)
    d_dense = Dense(VOCAB_SIZE, activation='softmax')
    d_outputs = d_dense(d_output)

    model = Model([e_input, d_input], d_outputs)

######################################### Decode #########################################

    enc_model = Model([e_input], e_states)

    d_input_h = Input(shape=(lstm_layers,))
    d_input_c = Input(shape=(lstm_layers,))
    d_states_inputs = [d_input_h, d_input_c]
    d_outputs, d_state_h, d_state_c = d_lstm(d_embeded, initial_state = d_states_inputs)
    d_states = [d_state_h, d_state_c]
    dec_model = Model([d_input] + d_states_inputs, [d_outputs] + d_states)


# ########################################################################

    enc_model.load_weights('/tmp/enc_model.h5')
    dec_model.load_weights('/tmp/dec_model.h5')
    model.load_weights('/tmp/main_model.h5')

    def input_sentence(text):
        user_ = clean_text(text)
        user = [user_]

        inp_sentence = []
        for sentence in user:
            lst = []
            for y in sentence.split():
                try:
                    lst.append(vocab[y])
                except:
                    lst.append(vocab['<OUT>'])
            inp_sentence.append(lst)
        
        inputs_sentence = pad_sequences(inp_sentence, max_input_len, padding='post')
        encoder_inputs = tensorflow.convert_to_tensor(inputs_sentence)
        states_value = enc_model.predict(encoder_inputs)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = vocab['<SOS>']

        return target_seq, states_value


################################# inference ##############################################

    user_ = inputs
    target_seq, states_value = input_sentence(user_)
    stop_condition = False
    decoded = ''

    while not stop_condition :
        output_tokens , h, c= dec_model.predict([target_seq] + states_value )
        input_tokens = d_dense(output_tokens)
        word_index = np.argmax(input_tokens)

        word = inv_vocab[word_index] + ' '
        if word != '<EOS> ':
            decoded += word  
        if word == '<EOS> ' or len(decoded.split()) > max_input_len:
            stop_condition = True 

        target_seq = np.zeros((1 , 1))  
        target_seq[0 , 0] = word_index
        states_value = [h, c]

    return decoded
    

#Codingan Trigger Chatbot from Input Messages
def hello_firestore(data, context):
    db = firestore.Client()
    #Read sumber path function ke trigger
    pathInput = data["value"]["name"]

    #Stored messagesId & userId berdasarkan trigger path
    messagesId = pathInput.split("/")[len(pathInput.split("/")) -1]
    userId = pathInput.split("/")[len(pathInput.split("/")) -3]

    #Read messages yang ada di document
    lastMsg = data["value"]["fields"]["messages"]["stringValue"]

    #Define path document yang akan di input response dari chatbot
    injected_doc = db.collection(u'chatbot').document(f'{userId}').collection(u'response').document(f'{messagesId}')
   
    #Process ML Chatbot
    responseMsg = hello_world(lastMsg)
    response = {
        u"messages": responseMsg,
        u"timestamp": str(datetime.now(pytz.timezone('Asia/Jakarta')))
    }

    #Write response ke path yg sudah ditentukan
    injected_doc.set(response)