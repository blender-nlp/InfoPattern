import csv
from tqdm import tqdm
import torch
import re
from torch.utils.data import DataLoader
from torch.optim import Adam
from IPython import embed
from langdetect import detect

from switching_lm.arguments import parse_args
from switching_lm.models.get_model import get_model
from switching_lm.utils import RunningMean


def remove_emojis(data):
    emoj = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def get_data(data_file):
    with open(data_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        lines = list(reader)
    train_data = []
    for text, label in lines[1:]:
        label = int(label) * 2 - 3
        language = detect(text)
        if language != "en":
            continue
        if ": " in text and "RT" in text[:10]:
            text = text[text.index(":")+2:]
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'#[a-zA-Z_]+', '', text)
        text = re.sub(r'@[a-zA-Z_]+', '', text)
        text = re.sub(r'  +', ' ', text)
        text = remove_emojis(text)
        text = re.sub(r'(# )*#', '', text)
        train_data.append({
            "text": text,
            "label": label
        })
    return train_data


def main(args):
    train_data = get_data(args.data_file)
    dataloader = DataLoader(
        train_data, batch_size=args.batch_size,
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
    loss_mean = RunningMean(0.95)
    scaler = torch.cuda.amp.GradScaler()

    for step_i in pbar:
        batch = next(data_iter, None)
        if batch is None:
            data_iter = iter(dataloader)
            batch = next(data_iter, None)

        batch_text = batch["text"]
        batch_stance = torch.Tensor(batch["label"]).to(device)
        tokenized = tokenizer(batch_text, padding=True,
                              max_length=args.max_length, truncation=True)
        input_ids = torch.LongTensor(tokenized["input_ids"]).to(device)

        optimizer.zero_grad()
        attention_mask = torch.LongTensor(tokenized["attention_mask"]).to(
            device)
        if args.low_resource_mode:
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16
            ):
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


if __name__ == "__main__":
    args = parse_args()
    main(args)
