import os
import json
import nltk
import math
import torch
import pickle
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import Counter, defaultdict
from argparse import ArgumentParser

def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('-src', '--src', type=str, required=True)
    parser.add_argument('-out', '--out', type=str, required=True)
    parser.add_argument('-sen', '--sen', type=str, default=None)
    parser.add_argument('-pre', '--pre', type=str, default=None)
    parser.add_argument('-output', '--output', type=str, default=None)
    parser.add_argument('-pre_save_path', '--pre_save_path', type=str, default='persona_pre.pkl')
    parser.add_argument('-train_src', '--train_src', type=str, default='../data-v2/persona/src-train.txt')
    parser.add_argument('-train_tgt', '--train_tgt', type=str, default='../data-v2/persona/tgt-train.txt')
    parser.add_argument('-src_vocab_path', '--src_vocab_path', type=str, default='../data-v2/persona/vocab.txt')
    parser.add_argument('-embedding_path', '--embedding_path', type=str, default='../glove.6B.300d.pkl')
    parser.add_argument('-embedding_size', '--embedding_size', type=int, default=300)
    opt = parser.parse_args()
    return opt

def read_data(file, split=True):
    with open(file, encoding='utf-8') as f:
        if split:
            data = [line.strip('\n').split(' ') for line in f]
        else:
            data = [line.strip('\n') for line in f]
    return data

def read_vocab(opt):
    vocab = {}
    with open(opt.src_vocab_path) as f:
        for i, line in enumerate(f):
            if i >= 50000:
                break
            
            word, cnt = line.strip('\n').split()
            if word in vocab:
                vocab[word] += int(cnt)
            else:
                vocab[word] = int(cnt)
    return vocab

def read_jsonlist(file):
    with open(file, encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def max_min_norm(value, maxi, mini):
    return (value - mini) / (maxi - mini)

def seq2vec(seq, vocab, word2idx):
    vec = np.zeros([len(vocab)], dtype=np.int)
    for item in set(seq):
        if item in word2idx:
            vec[word2idx[item]] = 1
    return vec

def to_class(val, classes_num, sort_dict, peer_num):
    for i in range(1, classes_num):
        if val > sort_dict[i * peer_num]:
            return i
    else:
        return classes_num

def spc_preprocess(train_src, train_tgt, vocab, pre_result):
    print('Start specificity-preprocess ... ')
    print('[SPC]: compute cnt_vec: ')
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    cnt_vec = np.zeros([len(vocab)], dtype=np.int)
    for item in tqdm(train_tgt):
        cnt_vec += seq2vec(item, vocab, word2idx)
    
    train_tgt_num = len(train_tgt)
    idf_vec = np.zeros([len(vocab)], dtype=np.int)
    for i in range(len(vocab)):
        if cnt_vec[i] != 0:
            idf_vec[i] = train_tgt_num / cnt_vec[i]

    print(f'[SPC]: max_count/data_num = {idf_vec.max() / train_tgt_num}')
    print(f'[SPC]: min_count/data_num = {idf_vec.min() / train_tgt_num}')
    print(f'[SPC]: avg_count/data_num = {idf_vec.mean() / train_tgt_num}')
    
    word2idf = {}
    for i in range(len(vocab)):
        if cnt_vec[i] != 0:
            idf = math.log(train_tgt_num / cnt_vec[i])
            word2idf[idx2word[i]] = idf

    max_idf = max(list(word2idf.values()))
    min_idf = min(list(word2idf.values()))

    word2idf_norm = {}
    for key, val in word2idf.items():
        word2idf_norm[key] = max_min_norm(val, maxi=max_idf, mini=min_idf)

    train_sen2spc = {}
    for i, rsp in tqdm(enumerate(train_tgt)):
        sentence_spc = []
        for word in rsp:
            if word not in word2idx:
                sentence_spc.append(0)
            else:
                sentence_spc.append(word2idf_norm[word])
        train_sen2spc[i] = sentence_spc

    idf_classes_num = 5
    sort_idf = sorted(list(chain(*train_sen2spc.values())), reverse=True)
    idf_peer_class_data_num = len(sort_idf) // idf_classes_num

    pre_result['Local']['SPC'] = {
        'word2idf_norm': word2idf_norm,
        'class_num': idf_classes_num,
        'peer_class_num': idf_peer_class_data_num,
        'sort_value_list': sort_idf,
        'train_sen2spc': train_sen2spc,
    }

    sen2idf = {}
    for item in tqdm(train_tgt):
        words = [token for token in item if token in word2idf_norm]
        if len(words) == 0:
            continue
        idf = sum([word2idf_norm[token] for token in words]) / len(words)
        sen2idf[' '.join(item)] = idf
    
    spe_classes_num = 3
    sort_sen_spe = sorted(list(sen2idf.values()), reverse=True)
    spe_peer_class_data_num = len(sort_sen_spe) // spe_classes_num

    pre_result['Global']['SPC'] = {
        'class_num': spe_classes_num,
        'peer_class_num': spe_peer_class_data_num,
        'sort_value_list': sort_sen_spe
    }

    pre_result['word2idx'] = word2idx
    pre_result['idx2word'] = idx2word
    
def read_embedding(path):
    with open(path, 'rb') as f:
        embedding = pickle.load(f)
    return embedding

def sentence_embedding_by_WR(sentence, pro_vocab, vec_vocab, alpha=1e-4):
    sent_emb = np.zeros([300])
    for word in sentence:
        if word in pro_vocab and word in vec_vocab:
            sent_emb += vec_vocab[word] * alpha / (alpha + pro_vocab[word])
    return sent_emb

def get_last_context(item):
    item = ' '.join(item).replace('<S2>', '<S1>')
    item = item.split('<S1>')[-1].strip().split(' ')
    return item

def cos_sim(word_emb, sent_emb):
    num = word_emb.dot(sent_emb.T)
    denom = np.linalg.norm(word_emb) * np.linalg.norm(sent_emb)
    return num / denom

def sim_preprocess(train_src, train_tgt, vocab, embedding_path, embedding_size, pre_result):
    print('Start similarity-preprocess ... ')
    print('[SIM]: load embedding: ')
    embedding = read_embedding(embedding_path)
    
    word_embeddings = torch.zeros([len(vocab), embedding_size])
    for i, word in enumerate(vocab):
        if word not in embedding:
            continue
        word_embeddings[i] = torch.from_numpy(embedding[word])
    
    pro_vocab = {key: value * 1.0 / len(vocab) for key, value in vocab.items()}
    
    word2idx = pre_result['word2idx']
    train_sent_emb = torch.zeros([len(train_src), embedding_size])
    for i, sen in tqdm(enumerate(train_src)):
        sen = get_last_context(sen)
        index = torch.LongTensor([word2idx[word] for word in sen if word in embedding and word in word2idx])
        weight = torch.Tensor([pro_vocab[word] for word in sen if word in embedding and word in word2idx]).view(1, -1)
        assert index.size(0) == weight.size(1)
        train_sent_emb[i] = torch.matmul(weight, word_embeddings.index_select(0, index))
    
    pre_result['train_sent_emb'] = train_sent_emb
    
    train_sen2sim = {}
    for i, rsp in tqdm(enumerate(train_tgt)):
        word_sim = []
        for word in rsp:
            if word not in embedding:
                word_sim.append(0)
            else:
                if train_sent_emb[i][0] == 0:
                    word_sim.append(0)
                else:
                    word_emb = embedding[word]
                    word_sim.append(cos_sim(word_emb, train_sent_emb[i]))
        train_sen2sim[i] = word_sim

    pre_result['train_sen2sim'] = train_sen2sim

    sort_sim_list = sorted(list(chain(*train_sen2sim.values())), reverse=True)

    sim_classes_num = 5
    sim_peer_class_data_num = len(sort_sim_list) // sim_classes_num

    pre_result['Local']['SIM'] = {
        'class_num': sim_classes_num,
        'peer_class_num': sim_peer_class_data_num,
        'sort_value_list': sort_sim_list
    }

    train_sen2sensim = {}
    for i, sim in tqdm(train_sen2sim.items()):
        try:
            train_sen2sensim[i] = sum(sim) / (len(sim) - sim.count(0))
        except:
            train_sen2sensim[i] = 3

    pre_result['train_sen2sensim'] = train_sen2sensim

    sort_sen2sensim = sorted(list(train_sen2sensim.values()), reverse=True)
    sensim_classes_num = 3
    sensim_peer_class_data_num = len(sort_sen2sensim) // sensim_classes_num

    pre_result['Global']['SIM'] = {
        'class_num': sensim_classes_num,
        'peer_class_num': sensim_peer_class_data_num,
        'sort_value_list': sort_sen2sensim
    }

    pre_result['pro_vocab'] = pro_vocab
    pre_result['embedding'] = embedding
    pre_result['word_embeddings'] = word_embeddings
    pre_result['embedding_size'] = embedding_size

def preprocess(opt):
    train_src = read_data(opt.train_src)
    train_tgt = read_data(opt.train_tgt)

    vocab = read_vocab(opt)

    pre_result = {
        'vocab': vocab,
        'Local': {},
        'Global': {}
    }

    spc_preprocess(train_src, train_tgt, vocab, pre_result)
    sim_preprocess(train_src, train_tgt, vocab, opt.embedding_path, opt.embedding_size, pre_result)

    pre_result['Global']['ASK'] = {
        'Keyword': ['?', '？', 'how', 'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why']}

    with open(opt.pre_save_path, 'wb') as f:
        pickle.dump(pre_result, f)
    print(f'Save pre-data at {opt.pre_save_path}')

def process(opt):
    assert os.path.exists(opt.pre)
    with open(opt.pre, 'rb') as f:
        pre_result = pickle.load(f)
    
    assert os.path.exists(opt.src), opt.src
    assert os.path.exists(opt.out), opt.out
    
    src_data = read_data(opt.src)
    out_data = read_data(opt.out)
    
    assert len(src_data) == len(out_data), f"{len(src_data)} != {len(out_data)}"
    
    out_attr = defaultdict(lambda: {'global': {}, 'local': {}})

    # spc process
    word2idf_norm = pre_result['Local']['SPC']['word2idf_norm']
    for i, item in enumerate(out_data):
        words = [token for token in item if token in word2idf_norm]
        if len(words) == 0:
            out_attr[i]['global']['spc'] = -1
        else:
            idf = sum([word2idf_norm[token] for token in words]) / len(words)
            out_attr[i]['global']['spc'] =  to_class(
                idf, 
                pre_result['Global']['SPC']['class_num'], 
                pre_result['Global']['SPC']['sort_value_list'], 
                pre_result['Global']['SPC']['peer_class_num'])

            out_attr[i]['local']['spc'] = [
                to_class(
                    word2idf_norm[word],
                    pre_result['Local']['SPC']['class_num'], 
                    pre_result['Local']['SPC']['sort_value_list'], 
                    pre_result['Local']['SPC']['peer_class_num']
                ) \
                if word in word2idf_norm else 0 \
                for word in item
            ] + [pre_result['Local']['SPC']['class_num']]
        out_attr[i]['global']['spc'] -= 1
    
    # sim process
    word2idx = pre_result['word2idx']
    embedding = pre_result['embedding']
    pro_vocab = pre_result['pro_vocab']
    word_embeddings = pre_result['word_embeddings']
    src_sen_emb = torch.zeros([len(src_data), pre_result['embedding_size']])
    for i, item in enumerate(src_data):
        sen = get_last_context(item)
        index = torch.LongTensor([word2idx[word] for word in sen if word in embedding and word in word2idx])
        weight = torch.Tensor([pro_vocab[word] for word in sen if word in embedding and word in word2idx]).view(1, -1)
        assert index.size(0) == weight.size(1)
        src_sen_emb[i] = torch.matmul(weight, word_embeddings.index_select(0, index))
    
    for i, item in enumerate(out_data):
        word_sim = []
        for word in item:
            if word not in embedding or src_sen_emb[i][0] == 0:
                word_sim.append(0)
            else:
                word_sim.append(cos_sim(embedding[word], src_sen_emb[i]))
        out_attr[i]['global']['sim'] = to_class(
            sum(word_sim) / len(word_sim),
            pre_result['Global']['SIM']['class_num'], 
            pre_result['Global']['SIM']['sort_value_list'], 
            pre_result['Global']['SIM']['peer_class_num']
            ) if len(word_sim) != 0 else -1
        out_attr[i]['global']['sim'] -= 1

        out_attr[i]['local']['sim'] = [
            to_class(
                val,
                pre_result['Local']['SIM']['class_num'], 
                pre_result['Local']['SIM']['sort_value_list'], 
                pre_result['Local']['SIM']['peer_class_num']
            ) - 1
            for val in word_sim
        ] + [pre_result['Local']['SIM']['class_num']]
    
    # len process
    for i, item in enumerate(out_data):
        length = len(item)
        if length < 10:
            sen_len = 1
        elif length >= 15:
            sen_len = 3
        else:
            sen_len = 2
        out_attr[i]['global']['len'] = sen_len
        out_attr[i]['global']['len'] -= 1
    
    # ask process
    for i, item in enumerate(out_data):
        out_attr[i]['global']['ask'] = 0
        for word in item:
            if word.lower() in pre_result['Global']['ASK']['Keyword']:
                out_attr[i]['global']['ask'] = 1
                break
    
    if opt.sen != None:
        assert os.path.exists(opt.sen)
        sen_data = read_data(opt.sen, split=False)
        for i, sen in enumerate(sen_data):
            out_attr[i]['global']['sen'] = int(sen)

    return out_attr

def main():
    opt = parse_opt()
    if opt.pre == None:
        preprocess(opt)
        opt.pre = opt.pre_save_path
    output = process(opt)
    out_file = opt.out + '.data_attr.txt' if opt.output == None else opt.output
    with open(out_file, 'w', encoding='utf-8') as f:
        for i in range(len(output)):
            f.write(str(output[i]) + '\n')

if __name__ == "__main__":
    main()