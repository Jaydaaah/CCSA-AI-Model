import torch
import json
from src.neural_model import NeuralNet
from src.intent_classes import Pattern, Tag
from src.nltk_utils import NLP_Util
import numpy as np

import os
import random


FOLDER_Path = "models/"
PROBLEM_PATH = "problems/"
EXTENSION = ".model"

I_DUNNO_RESPONSE = [
    "Sorry, I don't know the answer yet",
]

def GetLatestModel():
    models = [FOLDER_Path + file for file in os.listdir(FOLDER_Path) if file.endswith(EXTENSION)]
    return models[-1]


class ChatAI:
    no_answer = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 model_state: dict[str, any],
                 intents: list[dict[str, str | list[str]]],
                 bot_name: str = "AskAlly"):
        self.bot_name = bot_name
        self.model = NeuralNet(input_size,
                               hidden_size,
                               output_size).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval
        Tag.init_tags(intents)
        
    def response(self, msg, rephrase_repeat = True) -> tuple[str, bool]:
        pattern = Pattern(msg, record_stem = False)
        np_pattern = np.array(pattern.in_bag_words)
        np_pattern = np_pattern.reshape(1, np_pattern.shape[0])
        
        tensor_pattern = torch.from_numpy(np_pattern).to(self.device)
        
        output = self.model(tensor_pattern)
        _, predicted = torch.max(output, dim=1)
        
        tag = Tag.get_tag(int(predicted.item()))
        
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            return tag.execute_response(), True
        return self.idunno_response(msg), False
    
    def idunno_response(self, msg_input: str) -> str:
        text_path = os.path.join(PROBLEM_PATH, "No Answer Inputs.txt")
        with open(text_path, "a") as txtfile:
            txtfile.write("\n-> " + msg_input)
        return random.choice(I_DUNNO_RESPONSE)
    

    

if __name__ == "__main__":
    data = torch.load(GetLatestModel())
    Chat = ChatAI(data["input_size"],
                  data["hidden_size"],
                  data["output_size"],
                  data["model_state"],
                  data["intents"]
                  )
    print(f"Hi I'm {ChatAI.bot_name}, Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = Chat.response(sentence)
        print(resp)