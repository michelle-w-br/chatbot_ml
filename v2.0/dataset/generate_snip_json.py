from urllib.request import urlretrieve
from pathlib import Path
import json

SNIPS_DATA_BASE_URL = (
    "https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/master/data/snips/"
)
for filename in ["train", "valid", "test", "vocab.intent"]:
    path = Path(filename)
    if not path.exists():
        print(f"Downloading {filename}...")
        urlretrieve(SNIPS_DATA_BASE_URL + filename + "?raw=true", path)

def prepare_json(file_name):
    data = Path(file_name).read_text("utf-8").strip().splitlines()
    data_formated={}
    for line in data:
        setence, intent_label = line.split(" <=> ")
        word_with_label = setence.split()
        words = [item.rsplit(":", 1)[0]for item in word_with_label]
        #word_labels = [item.rsplit(":", 1)[1]for item in word_with_label]
        sentence=" ".join(words)
        if intent_label not in data_formated.keys():
            data_formated[intent_label]={}
            data_formated[intent_label]['patterns']=[]
        data_formated[intent_label]['patterns'].append(sentence)
    return data_formated

def save_json(file_name, formated_data):
   with open(file_name, 'w') as json_file:
      json.dump(formated_data, json_file, indent = 4, sort_keys=True)

train_data_formated=prepare_json("train")
valid_data_formated=prepare_json("valid")
test_data_formated=prepare_json("test")
save_json('snips_train.json',train_data_formated)
print("saved snips_train.json")
save_json('snips_valid.json',valid_data_formated)
print("saved snips_valid.json")
save_json('snips_test.json',test_data_formated)
print("saved snips_test.json")