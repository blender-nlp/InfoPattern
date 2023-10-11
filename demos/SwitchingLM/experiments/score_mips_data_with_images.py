import os
import glob
import json
import pickle
from tqdm import tqdm
import torch
from collections import Counter
from IPython import embed

from switching_lm.arguments import parse_args
from switching_lm.models.get_model import get_model


stance_types = [
    "Far Left", "Left", "Lean Left", "Center", "Lean Right", "Right",
    "Far Right"]


theme_conversion = dict(zip(
    [
        'Anti-Russian T-Shirt',
        'Anti-War Protests in Prague',
        'Merkel on Minsk Agreement',
        'Theme 1 Russophobia (The Claim that World is Unfair to Russia)',
        'Theme 2 Russian Slogans',
        'Theme 3 Ukraine and Nazi Claims',
        'Theme 4 Minsk Accords',
        'Theme 5 Attacking Zelensky',
        'Theme 6 NATO Exploited Ukraine',
        'Theme 7 US Fights Proxy War',
        'Theme 8 BRICS Superiority Claims',
        'Ukraine Bots',
        'Zelensky Bans Church',
        'Zelensky’s Anti-Union Law'
    ],
    [
        '37. Anti-Russian T-Shirt',
        '33. Anti-War Protests in Prague',
        '36. Merkel on Minsk Agreement',
        '24. Theme 1_ Russophobia (The Claim that World is Unfair to Russia)',
        '25. Theme 2_ Russian Slogans',
        '26. Theme 3_ Ukraine and Nazi Claims',
        '27. Theme 4_ Minsk Accords',
        '28. Theme 5_ Attacking Zelensky',
        '29. Theme 6_ NATO Exploited Ukraine',
        '30. Theme 7_ US Fights Proxy War',
        '31. Theme 8_ BRICS Superiority Claims',
        '34. Ukraine Bots',
        '35. Zelensky Bans Church',
        '32. Zelenskyбпs Anti-Union Law',
    ]
))


def main(args):
    with open(args.input_file, "rb") as f:
        tweets_with_images = pickle.load(f)

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model, tokenizer = get_model(
        args.model_name, args.adapted_component, 1,
        args.rank, args.epsilon, args.init_var)
    model.to_device(device)

    assert args.ckpt_name is not None
    model.load_state_dict(torch.load(args.ckpt_name)[1])

    all_results = {}
    for tweet_id, filename in tqdm(tweets_with_images.items()):
        theme, date = filename.split("/")[-2:]
        theme = theme_conversion[theme]
        date = date.split(".")[0]
        filename = glob.glob(f"{args.data_dir}/{theme}/*/{date}_tweets.jsonl")
        assert len(filename) == 1
        filename = filename[0]

        with open(filename, "r") as f:
            single_day = list(map(json.loads, f.readlines()))
        tweet = list(filter(lambda x: x["id"] == tweet_id,
                            single_day))[0]
        tweet_text = tweet["text"]

        if len(tweet_text) <= 3:
            best_guess, dist_list, token_evidence_list = None, None, None
        else:
            with torch.no_grad():
                best_guess_value, dist_list, token_evidence_list = \
                    model.switch_analysis(
                        tweet_text, 0
                    )
            best_guess = stance_types[int(best_guess_value) + 3]
            dist_list = dict(zip(stance_types, [_x[1] for _x in dist_list]))
            if best_guess != "Center":
                token_evidence_list = list(map(
                    lambda x: x[0],
                    filter(
                        lambda x: x[1] >= 0.8,
                        token_evidence_list,
                    )
                ))
            else:
                token_evidence_list = None
        all_results[tweet_id] = {
            "id": tweet["id"],
            "author_id": tweet["author_id"],
            "conversation_id": tweet["conversation_id"],
            "text": tweet_text,
            "best_guess": best_guess,
            "dist_list": dist_list,
            "token_evidence_list": token_evidence_list,
        }

    with open(args.output_file, "wb") as f:
        pickle.dump(all_results, f)
    embed()


if __name__ == "__main__":
    args = parse_args()
    main(args)
