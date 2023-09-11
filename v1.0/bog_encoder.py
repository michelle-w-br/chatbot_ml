import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

class BOW_Encoder():

    def __init__(self,sentences):
        self.vocab = set()
        self.init_vocab(sentences)

    def init_vocab(self, sentences):
        """
        build vocabulary from the sentences
        examples:
        sentences = ["hello morning!!", "I like apple", "hello world"]
        vocab = ["hello","morning","I","like","apple", "world"]
        """
        all_words=[]
        for sentence in sentences:
            words=nltk.word_tokenize(sentence)
            all_words.extend(words)
        stop_words=['?','.','!']
        all_words = [self.stem(w) for w in all_words if w not in stop_words]
        self.vocab=sorted(set(all_words))
        self.vocab_size=len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def stem(self, word):
        """
        find the root form of the word
        examples:
        words = ["start", "started", "starting"]
        [stem(w) for w in words]=["start","start","start"]
        """
        return stemmer.stem(word.lower())

    def encode(self, input):
        """
        return bag of words vector for the vocab
        example:
        input    = "hello, world"
        vocab    = ["hello","morning","I","like","apple", "world"]
        bog      = [  1 ,    0 ,    0 ,   0 ,    0 ,    1]
        """
        # stem each word
        words = nltk.word_tokenize(input) #word tokenize
        words = [self.stem(word) for word in words] #stem word
        embedding = np.zeros(self.vocab_size, dtype=np.float32)
        
        for idx, w in enumerate(self.vocab):
            if w in words: 
                embedding[idx] = 1

        return  embedding