from src.nltk_utils import lang, NLP_Util
import numpy as np
import random

class Tag:
    tags_list: list["Tag"] = []
    def __init__(self, tag: str, response: str | list[str], *args: str) -> None:
        # properties
        self.text = tag
        self.response = response
        self.patterns: list[Pattern] = []
        
        # create Pattern instances
        for arg in args:
            pat = Pattern(arg)
            self.patterns.append(pat)
            
        self.tags_list.append(self)

    def __int__(self) -> int:
        return self.tags_list.index(self)
    
    def __str__(self) -> str:
        return self.text
    
    def execute_response(self):
        if type(self.response) is str:
            return self.response
        elif type(self.response) is list:
            return random.choice(self.response)
        raise NotImplementedError(f"Response type of : {type(self.response)} is not supported")
            
    @staticmethod
    def get_tag(tag: str | int) -> "Tag":
        if type(tag) is int:
            return Tag.tags_list[tag]
        elif type(tag) is str:
            for t in Tag.tags_list:
                if t.text == tag:
                    return t
        raise NotImplementedError(f"Error type({type(tag)}) with value {tag}")
    
    @staticmethod
    def all_tags() -> list[str]:
        return [tag.text for tag in Tag.tags_list]
        
    @staticmethod
    def init_tags(intents: list[dict[str, str | list[str]]]):
        Pattern.reset_stemmed_words()
        Tag.tags_list.clear()
        for intent in intents:
            Tag(
                intent["tag"],
                intent["response"],
                *intent["patterns"]
            )
                
class Pattern:
    stemmed_words: list[str] = [] # all words
    def __init__(self, pattern_text: str, record_stem = True) -> None:
        
        # Properties
        self.pattern_text = pattern_text
        self.tokenize = NLP_Util.tokenize(pattern_text)
        self.stemmed_tokenize = [NLP_Util.stem(word) for word in self.tokenize if word != ""]

        # update stemmed words / all words
        for stem in self.stemmed_tokenize:
            if stem not in self.stemmed_words and record_stem:
                self.stemmed_words.append(stem)
        
    def __str__(self) -> str:
        return self.pattern_text
    
    @property
    def stemmed(self) -> list[str]:
        return self.stemmed_tokenize

    @property
    def in_bag_words(self) -> list[int]:
        bag = NLP_Util.bag_words(self.stemmed_tokenize, self.stemmed_words)
        return np.array(bag, dtype=np.float32)
    
    @staticmethod
    def reset_stemmed_words() -> None:
        Pattern.stemmed_words.clear()
        
        
class Training_Data:
    X_train: np.ndarray
    Y_train: np.ndarray
    Stemmed_words: list[str]
    All_tags: list[Tag]
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, stemmed_words: list[str], all_tags: list[Tag]):
        self.X_train = x_train
        self.Y_train = y_train
        self.Stemmed_words = stemmed_words
        self.All_tags = all_tags



def Training_Data_XY(json_data: list[dict[str, str | list[str]]]) -> Training_Data:
    Tag.init_tags(json_data)
    
    X = []
    Y = []
    for tag in Tag.tags_list:
        for pattern in tag.patterns:
            X.append(pattern.in_bag_words)
            Y.append(int(tag))
    return Training_Data(
        np.array(X),
        np.array(Y),
        Pattern.stemmed_words,
        Tag.all_tags()
    )
