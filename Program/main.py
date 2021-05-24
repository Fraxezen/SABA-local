#Libraries
import numpy as np
from numpy.lib.function_base import gradient
import tensorflow as tf
import re
import time
import os
from tensorflow._api.v2.compat.v1.nn import rnn_cell
from tensorflow.keras import initializers
from tensorflow.python.client import session
from tensorflow.python.keras.engine import training
import tensorflow_addons as tfa
from tensorflow_addons.utils.types import Optimizer


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
tf.compat.v1.disable_eager_execution()
def model_inputs():
    inputs = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None], name = 'input')
    targets = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None], name = 'target')
    lr = tf.compat.v1.placeholder(tf.compat.v1.int32, name = 'learning_rate')
    keep_prob = tf.compat.v1.placeholder(tf.compat.v1.int32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob
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
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                                cell_bw = encoder_cell, 
                                                                sequence_length = sequence_length, 
                                                                inputs = rnn_inputs, 
                                                                dtypes = tf.float32)
    return encoder_state


### Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tfa.seq2seq.AttentionWrapper(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tfa.seq2seq.BeamSearchDecoder(encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, name ='attn_dec_train')

    decoder_ouput, _, _, = tfa.seq2seq.dynamic_decode(decoder_cell, training_decoder_function, decoder_embedded_input, sequence_length, scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_ouput, keep_prob)
    return output_function(decoder_output_dropout)

### Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embedded_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tfa.seq2seq.AttentionWrapper(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    test_decoder_function = tfa.seq2seq.InferenceSampler(output_function, attention_keys, attention_values, attention_score_function, attention_construct_function, decoder_embedded_matrix, sos_id, eos_id, maximum_length, num_words, name ='attn_dec_inf')

    test_predictions, _, _, = tfa.seq2seq.dynamic_decode(decoder_cell, test_decoder_function, scope = decoding_scope)
    return test_predictions

### Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_creator_scope('decoding') as decoding_scope:
        lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.keras.initializers.TruncatedNormal(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.keras.layers.Dense(x, num_words, None, scope = decoding_scope, weights_initializer = weights, bias_initializer = biases)
        training_predition = decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size)

        decoding_scope.reuse_variable()
        test_predictions = decode_test_set(encoder_state,
                                            decoder_cell,
                                            decoder_embeddings_matrix,
                                            word2int['<SOS>'],
                                            word2int['<EOS>'],
                                            sequence_length-1,
                                            num_words,
                                            decoding_scope,
                                            output_function,
                                            keep_prob,
                                            batch_size)

    return training_predition, test_predictions


### Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, question_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionwords2int):
    encoder_embedded_input = tf.keras.layers.Embedding(inputs,
    answers_num_words + 1,
    encoder_embedding_size,
    initializers = tf.random_uniform_initializer(0,1))

    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_target = preprocess_target(targets, questionwords2int, batch_size)
    decoder_embedding_matrix = tf.Variable(tf.random.uniform([question_num_words+1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.compat.v1.nn.embedding_lookup(decoder_embedding_matrix, preprocessed_target)
    training_prediction, test_predictions = decoder_rnn( decoder_embedded_input, decoder_embedding_matrix, encoder_state, question_num_words, sequence_length, rnn_size, num_layers, questionwords2int, keep_prob, batch_size)

    return training_prediction, test_predictions


######### TRAINING SEQ2SEQ MODEL ############

# Setting the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layer = 3
encoding_embedding_size = 512
dencoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining session
tf.compat.v1.reset_default_graph()
session = tf.compat.v1.InteractiveSession()

# Load Model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting Seq Length
sequence_length = tf.compat.v1.placeholder_with_default(25, None, name = 'sequence_length')

# Getting the shape of inputs
input_shape = tf.shape(inputs)

# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                        targets,
                                                        keep_prob,
                                                        batch_size,
                                                        sequence_length,
                                                        len(answersword2int),
                                                        len(questionsword2int),
                                                        encoding_embedding_size,
                                                        dencoding_embedding_size,
                                                        rnn_size,
                                                        num_layer,
                                                        questionsword2int)


# Setting up loss Error, the Optimizer and gradient Clipping
with tf.name_scope('optimization'):
    loss_error = tfa.seq2seq.sequence_loss(training_predictions,
    targets,
    tf.compat.v1.ones([input_shape[0], sequence_length]))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    grads =  optimizer.compute_gradients(loss_error)
    clipped_gradients =  [(tf.compat.v1.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in grads if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)


# Padding the sequences with the <PAD> token
# Question: ['who', 'are', 'you', <PAD>, <PAD>, <PAD>, <PAD>]
# Answer:   [<SOS>, 'I', 'am', 'a', 'bot' '.', <EOS>, <PAD>]

def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

# Splitting the data into batches
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions)//batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answer_in_batch = answers[start_index : start_index + batch_size]
        padded_question_in_batch = np.array(apply_padding(questions_in_batch, questionsword2int))
        padded_answers_in_batch = np.array(apply_padding(answer_in_batch, answersword2int))
        yield padded_question_in_batch, padded_answers_in_batch

# Splitting the question and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions)) * 0.15
training_question = sorted_clean_questions[training_validation_split:]
training_question = sorted_clean_answers[training_validation_split:]
validation_question = sorted_clean_questions[:training_validation_split]
validation_question = sorted_clean_answers[:training_validation_split]

