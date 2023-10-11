import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn.functional as F
from torch.optim import Adam
from transformers import BertTokenizer, BertModel
from IPython import embed

from switching_lm.arguments import parse_args
from switching_lm.models.get_model import get_model
from switching_lm.utils import RunningMean


def get_data(data_dir):
    dataset = load_dataset("bavard/personachat_truecased")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    return train_data, val_data


def encode_profile_averaged(encoder, tokenizer, profile, device):
    with torch.no_grad():
        all_encoded = []
        for person in profile:
            tokenized = tokenizer(
                person, padding=True,
                max_length=args.max_length, truncation=True,
                return_tensors="pt"
            )
            tokenized = {_key: _v.to(device) for _key, _v in tokenized.items()}
            encoded = encoder(**tokenized).pooler_output.mean(0)
            all_encoded.append(encoded)
    all_encoded = torch.stack(all_encoded)
    return all_encoded


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


def main(args):
    train_data, val_data = get_data(args.data_dir)
    num_train_data = train_data.num_rows
    dataloader = DataLoader(
        range(num_train_data), batch_size=1,
        shuffle=True)
    data_iter = iter(dataloader)

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    profile_encoder = BertModel.from_pretrained("bert-base-uncased")
    profile_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    profile_encoder.train(False)
    profile_encoder.to(device)
    embedding_dim = profile_encoder.embeddings.word_embeddings.embedding_dim
    model, tokenizer = get_model(
        args.model_name, args.adapted_component, args.num_switches,
        args.rank, args.epsilon, args.init_var, embedding_dim)
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

    pbar = tqdm(range(start_step, args.n_steps))
    loss_mean = RunningMean(0.99)

    for step_i in pbar:
        optimizer.zero_grad()
        batch = next(data_iter, None)
        if batch is None:
            data_iter = iter(dataloader)
            batch = next(data_iter, None)

        datum = train_data[batch[0].item()]
        batch_inputs = datum["history"]
        batch_targets = datum["candidates"]
        batch_profile = datum["personality"]
        profile_encoded = encode_profile_averaged(
            profile_encoder, profile_tokenizer, [batch_profile], device
        )

        eos_token = tokenizer.eos_token
        batch_texts = [
            eos_token.join(batch_inputs) + eos_token + _this_target
            for _this_target in batch_targets
        ]
        target_pos = get_encoding_lengths(
            tokenizer, [eos_token.join(batch_inputs)]
        )[0]
        tokenized = tokenizer(
            batch_texts, padding=True,
            max_length=args.max_length, truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        logit_mask = attention_mask.clone()
        logit_mask[:, :target_pos+1] = 0

        logits = model(
            input_ids, attention_mask,
            profile_encoded
        ).logits
        batch_size, _, vocab_size = logits.shape
        token_nll = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            input_ids[:, 1:].reshape(-1), reduction="none"
        ).reshape(batch_size, -1)
        sentence_nll = (token_nll * logit_mask[:, :-1]).sum(1) / \
            logit_mask[:, :-1].sum(1)
        loss = (sentence_nll[-1] - sentence_nll[:-1] + 0.5).clamp(min=0)
        loss = loss[loss.isnan().logical_not()].mean()
        if loss.isnan().item():
            continue
        regularization_term = model.regularization_term()
        (loss + args.regularization * regularization_term).backward()
        optimizer.step()

        loss_mean.update(loss)
        pbar.set_description(
            f"{loss_mean.value}, {regularization_term.item()}")
        if (step_i+1) % args.log_step == 0:
            print(pbar.desc, flush=True)
            torch.save([step_i, args, model.state_dict()], args.ckpt_name)

    if not args.eval_only and args.ckpt_name is not None:
        torch.save([step_i, args, model.state_dict()], args.ckpt_name)

    def generate_on_profile(
        prompt, profile,
        min_length=20, max_length=1000, seed=None, resuming=True
    ):
        tokenized = profile_tokenizer(
            profile, padding=True, truncation=True,
            return_tensors="pt"
        )
        tokenized = {_key: _v.to(device) for _key, _v in tokenized.items()}
        profile_encoded = profile_encoder(**tokenized).pooler_output.mean(0)
        return model.generate(
            prompt, profile_encoded, device,
            min_length, max_length, seed, resuming)

    embed()


if __name__ == "__main__":
    args = parse_args()
    main(args)
