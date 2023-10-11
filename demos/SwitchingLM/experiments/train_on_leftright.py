import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from IPython import embed

from switching_lm.arguments import parse_args
from switching_lm.models.get_model import get_model
from switching_lm.utils import RunningMean


stances = ["Far Left", "Left", "Lean Left", "Center", "Lean Right", "Right",
           "Far Right"]


def get_data(data_dir):
    all_text = []
    all_stance = []
    all_topic = []

    for topic in os.listdir(data_dir):
        files = os.listdir(f"{data_dir}/{topic}/")
        for _file in files:
            if _file == "0-logs.json":
                continue
            with open(f"{data_dir}/{topic}/{_file}", "r") as f:
                filedata = json.load(f)
            for doc in filedata:
                if doc["maintext"] is None or doc["bias"] == "Unknown":
                    continue
                texts = [doc["title"]] + doc["maintext"].split("\n")
                for sent in texts:
                    all_text.append(sent)
                    all_stance.append(stances.index(doc["bias"]))

    return all_text, all_stance, all_topic


def main(args):
    all_text, all_stance, all_topic = get_data(args.data_dir)
    num_data = len(all_text)
    dataloader = DataLoader(
        range(num_data), batch_size=args.batch_size,
        shuffle=True)
    data_iter = iter(dataloader)

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model, tokenizer = get_model(
        args.model_name, args.adapted_component, 1,
        args.rank, args.epsilon, args.init_var,
        low_resource_mode=args.low_resource_mode)
    model.to_device(device)

    if args.eval_only:
        assert args.ckpt_name is not None
        model.load_state_dict(torch.load(args.ckpt_name)[1])
    if args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    if args.eval_only:
        args.n_steps = 0

    pbar = tqdm(range(args.n_steps))
    loss_mean = RunningMean(args.gamma_mean)
    scaler = torch.cuda.amp.GradScaler()

    for step_i in pbar:
        batch = next(data_iter, None)
        if batch is None:
            data_iter = iter(dataloader)
            batch = next(data_iter, None)

        batch_text = [all_text[i] for i in batch]
        batch_stance = torch.LongTensor(
            [all_stance[i] - 3 for i in batch]
        ).to(device)
        tokenized = tokenizer(batch_text, padding=True,
                              max_length=args.max_length, truncation=True)
        input_ids = torch.LongTensor(tokenized["input_ids"]).to(device)

        optimizer.zero_grad()
        attention_mask = torch.LongTensor(tokenized["attention_mask"]).to(
            device)
        if args.low_resource_mode:
            with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16):
                loss = model(
                    input_ids, attention_mask,
                    batch_stance.float()[:, None]
                ).loss
                regularization_term = model.regularization_term()
            scaler.scale(loss + args.regularization * regularization_term
                         ).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(
                input_ids, attention_mask,
                batch_stance.float()[:, None]
            ).loss
            regularization_term = model.regularization_term()
            (loss + args.regularization * regularization_term).backward()
            optimizer.step()

        loss_mean.update(loss)
        pbar.set_description(
            f"{loss_mean.value}, {regularization_term.item()}")
        if (step_i+1) % args.log_step == 0:
            print(pbar.desc, flush=True)

    if not args.eval_only and args.ckpt_name is not None:
        torch.save([args, model.state_dict()], args.ckpt_name)
    embed()


def generate(ckpt_name, prompt, stance_value, topic_name,
             min_length=20, max_length=100, seed=None,
             device=torch.device("cuda:0")):
    args, state_dict = torch.load(ckpt_name)
    model, _ = get_model(
        args.model_name, args.adapted_component, 1,
        args.rank, args.epsilon, args.init_var, args.low_resource_mode)
    model.load_state_dict(state_dict)
    model.to_device(device)
    return model.generate(
        prompt,
        [stance_value],
        min_length, max_length, seed
    )


def analyze(ckpt_name, prompt,
            device=torch.device("cuda:0")):
    args, state_dict = torch.load(ckpt_name)
    model, _ = get_model(
        args.model_name, args.adapted_component, 1,
        args.rank, args.epsilon, args.init_var, args.low_resource_mode)
    model.load_state_dict(state_dict)
    model.to_device(device)
    return model.switch_analysis(
        prompt, 0
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
