from enum import Enum
import re
import numpy as np

import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
import cebstemmer.stemmer as CEB_stemmer

EN_STEMMER = PorterStemmer()

class lang(Enum):
    EN = 0
    TAG = 1
    CEB = 2

class NLP_Util:
    @staticmethod
    def tokenize(word: str) -> list[str]:
        return nltk.word_tokenize(NLP_Util.clean_text(word))
    
    @staticmethod
    def clean_text(text: str ):
        # Remove all non-alphanumeric characters except single quotes and hyphens
        cleaned_text = re.sub(r"[^\sa-zA-Z0-9'-]", '', text)
        return cleaned_text
    
    @staticmethod
    def find_synonyms(word: str, word_list: list[str]):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name()
                if synonym in word_list:
                    synonyms.add(synonym)
        return synonyms
    
    @staticmethod
    def stem(tokenize_word: str, language: lang = None) -> str:
        """Detects language and return the stemmed words
        Usage: stemmed_words = [stem(word) for word in tokenize_words]"""
        if language is lang.TAG:
            return NLP_Util.stem_tag(tokenize_word)
        elif language is lang.CEB:
            return NLP_Util.stem_ceb(tokenize_word)
        elif language is lang.EN:
            return NLP_Util.stem_en(tokenize_word)
        else:
            if wordnet.synsets(tokenize_word):
                return NLP_Util.stem_en(tokenize_word)
            return NLP_Util.stem_ceb(tokenize_word)
    
    @staticmethod
    def stem_en(word: str) -> str:
        return EN_STEMMER.stem(word)
    
    @staticmethod
    def stem_ceb(word: str) -> str:
        return CEB_stemmer.stem_word(word, as_object=True).root
    
    @staticmethod
    def stem_tag(word: str) -> str:
        raise NotImplementedError("Not yet supported")
    
    @staticmethod
    def bag_words(word_list: list[str], all_word: list[str]) -> np.ndarray:
        bag = np.zeros((len(all_word),), dtype=np.float32)
        for i, word in enumerate(all_word):
            bag[i] = float(word in word_list)
        return np.array(bag)
    

    
if __name__ == '__main__':
    bisaya_word = "Hi, Mupalit ko'g learnings?"
    token_word = NLP_Util.tokenize(bisaya_word)
    stem_word = [NLP_Util.stem(word) for word in token_word]
    stem_word.remove("")
    print(stem_word)