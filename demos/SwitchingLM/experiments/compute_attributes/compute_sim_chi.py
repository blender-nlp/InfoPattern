from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


def compute_sim(sent1, sent2):
    sentences = [sent1, sent2]

    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return F.cosine_similarity(
        sentence_embeddings[0], sentence_embeddings[1],
        0)


def encode_corpus(corpus, batch_size, device, max_length=256, silent=False):
    dataloader = DataLoader(corpus, batch_size=batch_size, shuffle=False)
    output = []
    model.to(device)
    iterator = tqdm(dataloader) if not silent else dataloader
    for batch in iterator:
        encoded_input = tokenizer(
            batch, padding=True, truncation=True,
            max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(
                model_output, encoded_input["attention_mask"])
            sentence_embeddings = sentence_embeddings.cpu()
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        output.append(sentence_embeddings)
    return torch.cat(output, 0)
