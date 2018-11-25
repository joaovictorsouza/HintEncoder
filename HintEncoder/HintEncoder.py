import pandas as pd
import numpy as np
import re

from dataset import Tokenizer

from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector, Activation, Dropout, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding

def get_data(file_path):
    data = pd.read_csv(file_path, encoding='latin1')
    questions = data.Question
    answers = data.Answer

    return questions, answers

def train():
    questions, answers = get_data('train.csv')
    question_maxlen = max(map(len, (x for x in questions)))
    answer_maxlen = max(map(len, (x for x in answers)))
    max_question_answer = max(question_maxlen, answer_maxlen)


    tokenizer = Tokenizer("vocabulary.txt", max_question_answer)
    vocab_size = tokenizer.vocab_size()

    x = np.zeros((len(questions), max_question_answer, vocab_size), dtype=np.bool)
    y = np.zeros((len(answers), max_question_answer, vocab_size), dtype=np.bool)
   
    questions_t = questions.apply(tokenizer.tokenize_text)
    answers_t = answers.apply(tokenizer.tokenize_text)


    for i, sentence in enumerate(questions_t): 
        x[i] = tokenizer.encode(sentence, max_question_answer, vocab_size)
    for i, sentence in enumerate(answers_t):
        y[i] = tokenizer.encode(sentence, max_question_answer, vocab_size)


    hidden_size = 1024
    batch_size = 2
    epochs = 200
    print('Hidden Size / Batch size / Epochs = {}, {}, {}'.format(hidden_size, batch_size, epochs))

    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    model.add(LSTM(hidden_size, input_shape=(max_question_answer, vocab_size)))
    # For the decoder's input, we repeat the encoded input for each time step
    model.add(RepeatVector(max_question_answer))
    # The decoder LSTM could be multiple layers stacked or a single layer
    for _ in range(1):
        model.add(LSTM(hidden_size, return_sequences=True))
    
    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.05)
    model.save('my_model.h5')

train()
    


