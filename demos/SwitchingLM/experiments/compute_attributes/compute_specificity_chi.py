import numpy as np
import json

with open("experiments/compute_attributes/word_specificity.json", "r") as f:
    word_specificty_dict = json.load(f)


def compute_specificity(sent):
    sent = sent.lower().split(" ")
    sent_spec_score = [word_specificty_dict[word]
                       for word in sent if word in word_specificty_dict]
    if len(sent_spec_score) != 0:
        score = np.mean(sent_spec_score)
    else:
        score = 0
    return score
