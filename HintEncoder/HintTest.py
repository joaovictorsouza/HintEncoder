import pandas as pd
import numpy as np
import re

from dataset import Tokenizer

from keras.models import Sequential, load_model
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

questions, answers = get_data('test.csv')
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

model = load_model('my_model.h5')

for iteration in range(1, 200):
    print('-' * 50)
    print('Iteration', iteration)
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(2):
        ind = np.random.randint(0, len(x))
        rowX, rowy = x[np.array([ind])], y[np.array([ind])]
        preds = model.predict(rowX, verbose=0)
        q = tokenizer.decode(rowX[0])
        correct = tokenizer.decode(rowy[0])
        guess = tokenizer.decode(preds[0], calc_argmax=False)
        print('Q', q)
        print('T', correct)
        print('G',guess)
        print('---')
