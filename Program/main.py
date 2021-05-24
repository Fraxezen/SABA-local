#Libraries
import numpy as np
import tensorflow as tf
import re
import time
import os

from tensorflow.python.framework import dtypes


######### DATA PREPROCESSING ############

# Importing the dataset

lines = open(os.path.abspath('Program\\movie_lines.txt'), encoding = 'utf-8', errors= 'ignore').read().split('\n')
conversations = open(os.path.abspath('Program\\movie_conversations.txt'), encoding = 'utf-8', errors= 'ignore').read().split('\n')

# Data cleaning id and line
id2line={}

for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Conversation data cleaning
conversation_ids = []

for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversation_ids.append(_conversation.split(","))

# Seperate answer and question
# Question text adalah 1 text sebelum pertanyaan
questions = []
answers = []

for conversation in conversation_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])


# Cleaning the text, to make it easier to train
# Using testing data bahasa inggris
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

# Cleaning question
clean_questions = []

for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning answer
clean_answers = []

for answer in answers:
    clean_answers.append(clean_text(answer))

# creating dictionary to count word number in the question and answer
# to make the training more effecient
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

# Embedding the word
threshold = 20
questionsword2int = {}
word_number = 0

for word, count in word2count.items():
    if count >= threshold:
        questionsword2int[word] = word_number
        word_number += 1 

answersword2int = {}
word_number = 0

for word, count in word2count.items():
    if count >= threshold:
        answersword2int[word] = count
        word_number += 1 

# Adding Tokens
# SOS = start of string
# OOV = out of vocabulary
# EOS = End of string
tokens = ['<PAD>', '<EOS>', '<OOV>', '<SOS>']

for token in tokens:
    questionsword2int[token] = len(questionsword2int) + 1
for token in tokens:
    answersword2int[token] = len(answersword2int) + 1

# Creating invers for the answer
# Membuat invers dari int ke kata-kata untuk jawaban robot
answersint2word = {w_i : w for w, w_i in answersword2int.items()}

# Adding the EOS token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translating coversation into integer to train
# Replacing filtered word by OOV
questions_to_int = []
for questions in clean_questions:
    ints = []
    for question in questions.split():
        if question not in questionsword2int:
            ints.append(questionsword2int['<OOV>'])
        else:
            ints.append(questionsword2int[question])
        questions_to_int.append(ints)

answers_to_int = []
for answers in clean_answers:
    ints = []
    for answer in answers.split():
        if answer not in answersword2int:
            ints.append(answersword2int['<OOV>'])
        else:
            ints.append(answersword2int[answer])
        answers_to_int.append(ints)

# Sorting question and answer by the length of question
# Padding it in the same length as we want to
# Make it easier to train and optimize it

sorted_clean_questions = []
sorted_clean_answers = []

# 25  here is parameter of the longest question we want to have
for length in range(1, 25 + 1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])


########## BUILDING SEQ2SEQ MODEL ########

### Place holders for the iunput and target
# From numpy array data to tensor
# Tensor all variable must be define as tensor flow place holder
def model_input():
    inputs = tf.compat.v1.placeholder(tf.int32, [None, None], name = 'input')
    target = tf.compat.v1.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.compat.v1.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name = 'keep_prob')
    return input, target, lr, keep_prob
# [None, None] = 2 dimensional matrix
# keep_prob control dropout


### Preprocessing the target
# Batching
# Add SOS and delete the last token
def preprocess_target(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_target = tf.concat([left_side, right_side], 1)
    return preprocessed_target


### Encoder RNN Layer
# rnn_inputs = the model inputs like inpputs, target, lr
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                                cell_bw = encoder_cell, 
                                                                sequence_length = sequence_length, 
                                                                inputs = rnn_inputs, 
                                                                dtypes = tf.float32)
    return encoder_state