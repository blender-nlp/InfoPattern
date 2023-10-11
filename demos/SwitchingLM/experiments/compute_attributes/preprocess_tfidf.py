import json
import numpy as np
import collections
from transformers import GPT2Tokenizer


def keep_word(word):
    if ".com" in word:
        return False
    for ch in "abcdefghijklmnopqrstuvwxyz":
        if ch in word:
            return True
    return False


def gen_word_specificty():
    R = len(all_response)
    idf_freq_dict = {}
    for response in all_response:
        response_split = [word for word in response.split(" ") if keep_word(word)]
        response_split_word = set(response_split)
        for word in response_split_word:
            if word in idf_freq_dict:
                idf_freq_dict[word]["idf"] += 1
            else:
                idf_freq_dict[word] = {"freq": 0, "idf": 1}
        for word in response_split:
            idf_freq_dict[word]["freq"] += 1

    # get vocab
    word_freq_dict = {}
    for word in idf_freq_dict:
        word_freq_dict[word] = idf_freq_dict[word]["freq"]
    word_freq_dict = collections.Counter(word_freq_dict).most_common(100000)
    vocab = [elem[0] for elem in word_freq_dict]

    # compute specifity
    word_idf_dict = {}
    for word in vocab:
        word_idf_dict[word] = np.log(R / idf_freq_dict[word]["idf"])
    temp = [word_idf_dict[key] for key in word_idf_dict]
    min_idf = min(temp)
    max_idf = max(temp)
    print("max idf: {}, min idf: {}".format(max_idf, min_idf))
    json.dump(word_idf_dict, open("word_idf.json", "w"), ensure_ascii=False, indent=2)

    nidf_dict = {}
    for word in word_idf_dict:
        nidf_dict[word] = (word_idf_dict[word] - min_idf) / (max_idf - min_idf)  # min-max normalization
    f_w = open("word_specificty.json", "w")
    json.dump(nidf_dict, f_w, ensure_ascii=False, indent=2)


def compute_specificty(sent, word_specificty_dict):
    sent = sent.lower().split(" ")
    sent_spec_score = [word_specificty_dict[word] for word in sent if word in word_specificty_dict]
    if len(sent_spec_score) != 0:
        score = np.mean(sent_spec_score)
    else:
        score = 0
    return score


def compute_length(sent, tokenizer):
    sent_id = tokenizer.tokenize(sent)
    return len(sent_id)


def compute_qa(sent):
    question_marker = ["?", "？", "how", "what", "when", "where", "which", "who", "whom", "whose", "why"]
    is_question = False
    for marker in question_marker:
        if marker in sent:
            is_question = True
            break
    return is_question


if __name__ == "__main__":
    gen_word_specificty()
   
    word_specificty_dict = json.load(open("word_specificty.json"))
    tokenizer = GPT2Tokenizer.from_pretrained("/mm/huzhe01/text-gen/GPT2-seq2seq/model_card/gpt2-pytorch")
    print("finish loading")
    
    data_path = "../train-process.jsonl"
    f_w = open("train-process-spec.jsonl", "w")    
    f_r = open(data_path)
    line = f_r.readline()
    index = 0
    while line:
        index += 1
        if index % 100000 == 0:
            print(index)
        line_json = json.loads(line)
        # {"id": 0, "tid": "t3_17830,t1_c24,t1_c40", "context": "On the bright side , despite kidnapping and cruelly abandoning him , it doesn't sound like he was tortured ...", "response": "We didn't torture somebody ! USA"}
        context_list = line_json["context"].split("<#EOS#>")
        context_list = [elem.lower() for elem in context_list]       
        response = line_json["response"].lower()
        context_spec_list = [compute_specificty(utter, word_specificty_dict) for utter in context_list]
        context_len_list = [compute_length(utter, tokenizer) for utter in context_list]
        context_qa_list = [compute_qa(utter) for utter in context_list]
        response_spec = compute_specificty(response, word_specificty_dict)
        response_len = compute_length(response, tokenizer)
        response_qa = compute_qa(response)
        line_json["context_attribute"] = {"specificity": context_spec_list, "q-a": context_qa_list, "length": context_len_list}
        line_json["response_attribute"] = {"specificity": response_spec, "q-a": response_qa, "length": response_len}
        f_w.write(json.dumps(line_json) + "\n")
        line = f_r.readline()



