import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoTokenizer  # BertTokenizer, BertModel
from collections import defaultdict
from IPython import embed

from switching_lm.arguments import parse_args
from switching_lm.models.get_model import get_model
from switching_lm.utils import RunningMean


def get_data():
    dataset = load_dataset("bavard/personachat_truecased")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    return train_data, val_data


def get_encoding_lengths(tokenizer, texts):
    tokenized = tokenizer(
        texts, padding=True,
        max_length=args.max_length, truncation=True,
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"]
    length = input_ids.shape[1]
    target_pos = (
        (input_ids != tokenizer.eos_token_id) *
        torch.arange(0, length, 1, dtype=torch.int64)[None]
    ).max(1).values
    return target_pos


def preprocess_attributes(dataset, tokenizer, device):
    from experiments.compute_attributes.compute_specificity_chi import \
        compute_specificity
    from experiments.compute_attributes.preprocess_tfidf import \
        compute_qa, compute_length
    from experiments.compute_attributes.sentiment import compute_sentiment
    from experiments.compute_attributes.compute_sim_chi import \
        encode_corpus

    all_previous = [datum[-1] for datum in dataset["history"]]
    all_sentences = sorted(set([_sent for _group in dataset["candidates"] for
                                _sent in _group]))
    sent_to_id = dict(zip(all_sentences, range(len(all_sentences))))
    sent_attributes = defaultdict(dict)
    print("encoding history and candidates")
    encoded_previous = encode_corpus(all_previous, 256, device)
    encoded_sentences = encode_corpus(all_sentences, 256, device)
    print("computing other attributes")
    for _sent in tqdm(all_sentences):
        sent_attributes[_sent]["length"] = compute_length(_sent, tokenizer)
        sent_attributes[_sent]["qa"] = compute_qa(_sent)
        sent_attributes[_sent]["sentiment"] = compute_sentiment(_sent)
        sent_attributes[_sent]["specificity"] = compute_specificity(
            _sent)
    print("reorganizing attributes")
    all_results = []
    for datum_i, datum in tqdm(enumerate(dataset), total=len(dataset)):
        output = []
        history_feature = encoded_previous[datum_i]
        for _cand in datum["candidates"]:
            cand_id = sent_to_id[_cand]
            cand_feature = encoded_sentences[cand_id]
            cand_attributes = sent_attributes[_cand]
            output.append([
                (cand_attributes["length"] - 12.5) / 3,
                cand_attributes["qa"] * 6 - 3,
                cand_attributes["sentiment"] * 3 - 3,
                cand_attributes["specificity"] * 15 - 5,
                F.cosine_similarity(
                    history_feature, cand_feature, 0
                ) * 12 - 3,
            ])
        output = np.array(output)
        all_results.append(output)
    return all_results


def compute_single_attribute(history, response, tokenizer, device):
    from experiments.compute_attributes.compute_specificity_chi import \
        compute_specificity
    from experiments.compute_attributes.preprocess_tfidf import \
        compute_qa, compute_length
    from experiments.compute_attributes.sentiment import compute_sentiment
    from experiments.compute_attributes.compute_sim_chi import \
        encode_corpus

    encoded_previous = encode_corpus([history], 256, device, True)
    encoded_sentences = encode_corpus([response], 256, device, True)
    return [
        (compute_length(response, tokenizer) - 12.5) / 3,
        compute_qa(response) * 6 - 3,
        compute_sentiment(response) * 3 - 3,
        compute_specificity(response) * 15 - 5,
        F.cosine_similarity(
            encoded_previous[0], encoded_sentences[0], 0
        ).item() * 12 - 3
    ]


def train(dataloader, dataset, dataset_attributes, start_step, args,
          model, tokenizer, optimizer, device):
    data_iter = iter(dataloader)
    pbar = tqdm(range(start_step, args.n_steps))
    loss_mean = RunningMean(0.99)

    for step_i in pbar:
        optimizer.zero_grad()
        batch = next(data_iter, None)
        if batch is None:
            data_iter = iter(dataloader)
            batch = next(data_iter, None)

        batch_data = [dataset[i.item()] for i in batch]
        attributes = torch.Tensor(
            np.stack([dataset_attributes[i.item()][-1] for i in batch])
        ).to(device)
        attributes = torch.cat(
            [attributes, torch.ones_like(attributes[:, 0, None]) * 10],
            1
        )
        batch_inputs = [_datum["history"] for _datum in batch_data]
        batch_targets = [_datum["candidates"][-1] for _datum in batch_data]

        eos_token = tokenizer.eos_token
        batch_texts = [
            eos_token.join(_this_input) + eos_token + _this_target
            for _this_input, _this_target in zip(batch_inputs, batch_targets)
        ]
        target_pos = get_encoding_lengths(
            tokenizer, [
                eos_token.join(_this_input)
                for _this_input in batch_inputs
            ]
        )
        tokenized = tokenizer(
            batch_texts, padding=True,
            max_length=args.max_length, truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        logit_mask = attention_mask.clone()
        for i, _p in enumerate(target_pos):
            logit_mask[i, :_p+1] = 0

        logits = model(
            input_ids, attention_mask,
            attributes
        ).logits
        batch_size, _, vocab_size = logits.shape
        token_nll = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            input_ids[:, 1:].reshape(-1), reduction="none"
        ).reshape(batch_size, -1)
        loss = token_nll[logit_mask[:, :-1].bool()].mean()
        regularization_term = model.regularization_term()
        (loss + args.regularization * regularization_term).backward()
        optimizer.step()

        loss_mean.update(loss)
        pbar.set_description(
            f"{loss_mean.value}, {regularization_term.item()}")
        if (step_i+1) % args.log_step == 0:
            print(pbar.desc, flush=True)
            torch.save([step_i, args, model.state_dict()], args.ckpt_name)


def evaluate(dataloader, dataset, dataset_attributes,
             args, model, tokenizer, device):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    num_data = dataset.num_rows
    scores = np.zeros((num_data, 5))
    ground_truth = np.zeros((num_data, 5))
    responses = []

    for datum_i, datum in tqdm(enumerate(dataset), total=num_data):
        datum_inputs = datum["history"]
        # datum_targets = datum["candidates"][-1]
        attributes = torch.Tensor(
            dataset_attributes[datum_i][-1].tolist() + [10]
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                datum_inputs, attributes, device,
                min_length=20, max_length=1024, seed=0,
            )

        evaluated_attributes = compute_single_attribute(
            datum_inputs[-1], output, tokenizer, device
        )
        scores[datum_i] = evaluated_attributes
        ground_truth[datum_i] = attributes[:5].cpu().numpy()
        responses.append(output)

    accuracy = (
        np.sign(scores) == np.sign(ground_truth)
    ).astype(float).mean(0)

    return scores, responses, accuracy


def main(args):
    train_data, val_data = get_data()
    num_train_data = train_data.num_rows
    num_val_data = val_data.num_rows
    train_dataloader = DataLoader(
        range(num_train_data), batch_size=args.batch_size,
        shuffle=True)
    val_dataloader = DataLoader(
        range(num_val_data), batch_size=args.batch_size,
        shuffle=False)

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    attribute_file = os.path.join(args.data_dir, "attributes.pt")
    if os.path.exists(attribute_file):
        print("loading attributes from", attribute_file)
        with open(attribute_file, "rb") as f:
            attribute_data = pickle.load(f)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        attribute_data = (
            preprocess_attributes(train_data, tokenizer, device),
            preprocess_attributes(val_data, tokenizer, device)
        )
        with open(attribute_file, "wb") as f:
            pickle.dump(attribute_data, f)
    # train_attributes = attribute_data[0]

    model, tokenizer = get_model(
        args.model_name, args.adapted_component, args.num_switches,
        args.rank, args.epsilon, args.init_var, 0)
    model.to_device(device)

    start_step = 0
    if args.eval_only or os.path.exists(args.ckpt_name):
        print("Loading from ckpt:", args.ckpt_name)
        ckpt = torch.load(args.ckpt_name)
        model.load_state_dict(ckpt[2])
        start_step = ckpt[0]
    if args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    if args.eval_only:
        args.n_steps = 0

    train(train_dataloader, train_data, attribute_data[0], start_step, args,
          model, tokenizer, optimizer, device)

    if not args.eval_only and args.ckpt_name is not None:
        torch.save([args.n_steps, args, model.state_dict()], args.ckpt_name)
    else:
        def generate_on_profile(
            prompt, switch_values,
            min_length=20, max_length=1000, seed=0, resuming=False
        ):
            return model.generate(
                prompt, switch_values, device,
                min_length, max_length, seed, resuming)
        embed()
        exit()
        evaluate(val_dataloader, val_data, attribute_data[1], args,
                 model, tokenizer, device)


if __name__ == "__main__":
    args = parse_args()
    main(args)
