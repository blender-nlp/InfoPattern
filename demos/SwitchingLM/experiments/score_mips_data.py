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


def get_data(data_dir):
    filelist = glob.glob(
        os.path.join(data_dir, "**/*_tweets.jsonl"),
        recursive=True)
    all_data = []
    all_text = []
    for _file in tqdm(filelist):
        with open(_file, "r") as f:
            this_data = list(map(json.loads, f.readlines()))
            all_data.append(this_data)
            all_text.append([_datum.get("text", "") for _datum in this_data])

    internal_filelist = [
        os.path.join(*_file.split("/")[-3:])
        for _file in filelist
    ]
    return internal_filelist, all_data, all_text


def main(args):
    # tweet data
    cache_file = os.path.join(args.data_dir,
                              "processed/data_loading_cache.pkl")
    if not os.path.exists(cache_file):
        filelist, all_data, all_text = get_data(args.data_dir)
        with open(cache_file, "wb") as f:
            print("dumping to data cache:", cache_file)
            pickle.dump([filelist, all_data, all_text], f)
    else:
        with open(cache_file, "rb") as f:
            print("loading from data cache:", cache_file)
            filelist, all_data, all_text = pickle.load(f)
            print("loading done")

    # user data
    user_data = []
    for _file in tqdm(filelist):
        with open(os.path.join(
                args.data_dir, _file.replace("tweets", "users")), "r") as f:
            user_data.append(list(map(json.loads, f.readlines())))

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model, tokenizer = get_model(
        args.model_name, args.adapted_component, 1,
        args.rank, args.epsilon, args.init_var)
    model.to_device(device)

    assert args.ckpt_name is not None
    model.load_state_dict(torch.load(args.ckpt_name)[1])

    all_results = []
    num_samples = 5
    for _file, _data, _texts in tqdm(zip(filelist, all_data, all_text),
                                     total=len(filelist)):
        file_result = []
        for _datum, _t in zip(_data[:num_samples], _texts[:num_samples]):
            if len(_t) <= 3:
                best_guess, dist_list, token_evidence_list = None, None, None
            else:
                with torch.no_grad():
                    best_guess_value, dist_list, token_evidence_list = \
                        model.switch_analysis(
                            _t, 0
                        )
                best_guess = stance_types[int(best_guess_value) + 3]
                dist_list = dict(zip(stance_types, [_x[1] for _x in dist_list]))
                if best_guess != "Center":
                    token_evidence_list = list(map(
                        lambda x: x[0],
                        filter(
                            lambda x: x[1] >= 1,
                            token_evidence_list,
                        )
                    ))
                else:
                    token_evidence_list = None
            file_result.append({
                "id": _datum["id"],
                "author_id": _datum["author_id"],
                "conversation_id": _datum["conversation_id"],
                "text": _t,
                "best_guess": best_guess,
                "dist_list": dist_list,
                "token_evidence_list": token_evidence_list,
            })
        all_results.append({
            "filename": _file,
            "analysis": file_result
        })
    result_file = os.path.join(args.data_dir, "stance_analysis.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=4)

    stances = []
    languages = []
    locations = []
    hashtags = []
    entities = []
    for _file, _data, _results, _users in tqdm(
        zip(filelist, all_data, all_results, user_data),
            total=len(filelist)):
        for _datum, _result, _user in zip(
            _data[:num_samples], _results["analysis"][:num_samples],
            _users[:num_samples]
        ):
            stances.append(stance_types.index(_result["best_guess"])-3 if
                           _result["best_guess"] is not None else 0)
            languages.append(_datum["lang"])
            locations.append(_user.get("location", None))
            if "entities" in _datum:
                _hashtag = _datum["entities"].get("hashtags", None)
                if _hashtag is not None:
                    _hashtag = [_h["tag"] for _h in _hashtag]
                _anno = _datum["entities"].get("annotations", None)
                if _anno is not None:
                    _anno = [_a["normalized_text"] for _a in _anno]
                hashtags.append(_hashtag)
                entities.append(_anno)
            else:
                hashtags.append(None)
                entities.append(None)

    counters = {}
    for _type in ["stances", "languages", "locations", "hashtags", "entities"]:
        counters[_type] = {"left": Counter(), "right": Counter()}
    for _fine_stance, _lang, _loc, _hashtags, _entities in zip(
        stances, languages, locations, hashtags, entities
    ):
        if _fine_stance == 0:
            continue
        _stance = "left" if _fine_stance < 0 else "right"
        counters["stances"][_stance].update([_fine_stance])
        counters["languages"][_stance].update([_lang])
        counters["locations"][_stance].update([_loc])
        counters["hashtags"][_stance].update(_hashtags)
        counters["entities"][_stance].update(_entities)


if __name__ == "__main__":
    args = parse_args()
    main(args)
