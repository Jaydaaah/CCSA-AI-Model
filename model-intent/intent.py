"""_summary_
tag = what is the intent of the question by the client
patterns = list of string/ questions from the client
response = the response of the chatbot to the client
"""

from src.train import Trainer
import json
import os
import time

os.system("cls")
working_directory = "model-intent/"
filename = working_directory + "intent.json"

with open(filename, "r") as json_file:
    data = json.load(json_file)
    Intents = data['intents']


# test if intent is properly configured
Tags = []
Patterns = []
for intent in Intents:
    tag = intent["tag"].lower()
    if tag in Tags:
        raise ValueError(f"🎉: Tag named: {tag} has duplicate")
    if type(intent["patterns"]) is not list:
        raise ValueError(f"🎉: Pattern at tag: {tag} is not an array. please enclose it with []")
    for pattern in intent["patterns"]:
        if pattern.lower() in Patterns:
            raise ValueError(f"🎉: Duplicate pattern name `{pattern}` at tag: `{tag}`")
        Patterns.append(pattern.lower())
    Tags.append(tag)
print("\n\n🙌: ***Intent is properly configured***")
print()


# start training using the configured intent
train = Trainer(Intents)
print("💪: start training...")
print("📝: the closer the Loss to 0.0 the more accurate it is")
train.start_train(print_output=True)
trained_model = train.get_model()
is_test_model = "yes" in input("\nDo you want to test the model? (yes/no) ").lower()

# testing model
if (is_test_model):
    print("\n")
    print("🧪: Testing Model...")
    print("📝: Type 'done' to proceed to next step\n")
    time.sleep(2)
    print("🤖: Hello there👋")
    while True:
            # sentence = "do you use credit cards?"
            sentence = input("You: ").strip()
            if sentence == "done":
                break
            resp, status = trained_model.response(sentence)
            print(f"🤖: {resp}")
        
# saving model
is_saving_model = "yes" in input("\nDo you want to save the model? (yes/no) ").lower()
if (is_saving_model):
    train.save_model()
    print("🙌: Model saved. Thank you for your effort...🎉🎈\n\n")
else:
    print("🙌: Thank youu. program exiting now...\n\n")