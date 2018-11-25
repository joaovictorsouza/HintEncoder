import string
import numpy as np

class Tokenizer:
    not_found_token = 'NF'

    def __init__(self, vocab_path, max_seq_len):
        self.vocab_path = vocab_path
        self.max_seq_len = max_seq_len
        self.word_to_index = {self.not_found_token : 0}
        self.index_to_word = {0 : self.not_found_token}
        self.__load_vocab(vocab_path)

    def __load_vocab(self, vocab_path):
        with open(vocab_path, encoding='utf-8') as f:
            for line in f:
                line = line[:-1]
                self.add_vocab_word(line)

    def remove_ponctuation(self, data):
        table = str.maketrans({key: None for key in string.punctuation})
        return data.translate(table)   

    def vocab_size(self):
        return len(self.word_to_index)

    def add_vocab_word(self, word):
        idx = len(self.word_to_index)
        self.word_to_index[word] = idx
        self.index_to_word[idx] = word

    def encode(self, sentence, maxlen, vocab_size):
        X = np.zeros((maxlen, vocab_size))
        for i, c in enumerate(sentence):
            try:
                X[i, self.word_to_index[ self.remove_ponctuation(c.lower())]] = 1
            except Exception as e:
                X[i, self.word_to_index[Tokenizer.not_found_token]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        data = []
        if calc_argmax:
            X = X.argmax(axis=-1)
        for x in X:
            if(self.index_to_word[x] != self.not_found_token):
                data.append(self.index_to_word[x])
        return ' '.join(data)


    def tokenize_text(self, paragraph):
        paragraph = self.remove_ponctuation(paragraph.lower())
        results = []
        for w in paragraph.split(' '):
            results.append(w)
        return results

    def text_to_sequence(self, paragraph):
        paragraph = self.remove_ponctuation(paragraph.lower())
        results = []
        for w in paragraph.split(' '):
            try:
                idx = self.word_to_index[w]
            except Exception as e:
                idx = self.word_to_index[Tokenizer.not_found_token]
                pass

            results.append(idx)


        # reverse and ensure we cut off or pad
        results = results[::-1][:self.max_seq_len]
        results.extend([0] * (self.max_seq_len - len(results)))
        return np.asarray(results)

    def texts_to_sequences(self, texts):
        results = []
        for text in texts:
            result = self.text_to_sequence(text)
            results.append(result)

        return np.asarray(results)

