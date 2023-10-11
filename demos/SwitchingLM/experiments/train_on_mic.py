import os
import csv
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from IPython import embed

from switching_lm.arguments import parse_args
from switching_lm.models.get_model import get_model
from switching_lm.utils import RunningMean


def get_data(data_file):
    with open(data_file, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    header = dict(zip(data[0], range(len(data[0]))))
    data = [
        {
            "Q": _item[header["Q"]],
            "A": _item[header["A"]],
            "label": int(_item[header["A_agrees"]])
        } for _item in data[1:] if len(_item) == 13
    ]
    print("dataset size:", len(data))
    return data


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


def train(dataloader, dataset, start_step, args,
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

        batch_inputs = batch["Q"]
        batch_targets = batch["A"]
        labels = torch.Tensor(
            batch["label"]
        )[:, None].to(device) - 1

        eos_token = tokenizer.eos_token
        batch_texts = [
            _this_input + eos_token + _this_target
            for _this_input, _this_target in zip(batch_inputs, batch_targets)
        ]
        target_pos = get_encoding_lengths(
            tokenizer, batch_inputs
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
            labels
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


def main(args):
    train_data = get_data(args.data_file)
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True)

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

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

    train(train_dataloader, train_data, start_step, args,
          model, tokenizer, optimizer, device)

    if not args.eval_only and args.ckpt_name is not None:
        torch.save([args.n_steps, args, model.state_dict()], args.ckpt_name)

    def generate_on_profile(
        prompt, switch_value,
        min_length=20, max_length=1000, seed=0, resuming=False,
        beam_search=False, beam_sample=False
    ):
        return model.generate(
            [prompt], torch.Tensor([switch_value]).to(device), device,
            min_length, max_length, seed, resuming, beam_search, beam_sample)
    embed()
    exit()


if __name__ == "__main__":
    args = parse_args()
    main(args)
