import gensim
import json
from STOPWORDS import stopwords
from sentence_transformers import SentenceTransformer, util
import time
import os
import sys

start, end = int(sys.argv[1]), int(sys.argv[2])

sent_bert = SentenceTransformer('/mm/huzhe01/model/paraphrase-MiniLM-L12-v2')
sent_bert.max_seq_length = 256
model = gensim.models.KeyedVectors.load_word2vec_format("/mnt/sg/huzhe01/workspace/Glove/word2vec_format.vec")
#model.wmdistance(["this", "is", "good"], ["nice", "food"])
print("finish loading model")

#start, end = 0, 10
all_file_list = os.listdir("../split")
file_list = []
for elem in all_file_list:
    elem_id = int(elem.replace("train-process-chunk_", "").replace(".jsonl", ""))
    if elem_id in range(start, end):
        file_list.append(elem)

print("current stat: {}, end: {}".format(start, end))
print("all files: ", file_list)

finished_file = os.listdir("split-sim")
for tfile in file_list:
    print(tfile)
    if tfile.replace(".jsonl", "-sim.jsonl") in finished_file:
        print(tfile, ": finished!")
        continue
    write_path = "split-sim/" + tfile.replace(".jsonl", "-sim.jsonl") 
    f_w = open(write_path, "w")
    indx = 0
    start_time = time.time()
    f_r = open("../split/" + tfile)
    line = f_r.readline()
    while line:
        if indx % 100000 == 0:
            end_time = time.time()
            print(indx, end_time - start_time)
            start_time = time.time()
        indx += 1

        line = json.loads(line)
        context = line["context"]
        response = line["response"]
        last_utter = context.split("<#EOS#>")[-1]

        cur_sim = model.wmdistance([elem for elem in last_utter.split(" ") if elem not in stopwords], 
                                   [elem for elem in response.split(" ") if elem not in stopwords])

        # compute sentence-bert
        embeddings1 = sent_bert.encode(last_utter, convert_to_tensor=True)
        embeddings2 = sent_bert.encode(response, convert_to_tensor=True)
        #Compute cosine-similarits
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2).tolist()[0]
        line["wmd_sim"] = cur_sim
        line["sent_bert_sim"] = cosine_scores
        f_w.write(json.dumps(line) + "\n")
        line = f_r.readline()

print("finished")


