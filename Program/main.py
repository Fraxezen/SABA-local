#libraries

import numpy as np
import tensorflow as tf
import re
import time
import os


######### Data Preprocessing ############

# importing the dataset

lines = open(os.path.abspath('Program\\movie_lines.txt'), encoding = 'utf-8', errors= 'ignore').read().split('\n')
conversations = open(os.path.abspath('Program\\movie_conversations.txt'), encoding = 'utf-8', errors= 'ignore').read().split('\n')

# data cleaning id and line
id2line={}

for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# conversation data cleaning

conversation_ids = []

for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversation_ids.append(_conversation.split(","))

print(len(conversation_ids[0]))