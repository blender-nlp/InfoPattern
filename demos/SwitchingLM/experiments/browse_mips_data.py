import os
import glob
import json
import numpy as np
from tqdm import tqdm
from IPython import embed

from switching_lm.arguments import parse_args


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
    filelist, all_data, all_text = get_data(args.data_dir)
    user_data = []
    for _file in tqdm(filelist):
        with open(os.path.join(
                args.data_dir, _file.replace("tweets", "users")), "r") as f:
            user_data.append(list(map(json.loads, f.readlines())))

    id_list = {}
    for _file_i, _group in enumerate(all_data):
        for _tweet_i, _tweet in enumerate(_group):
            id_list[_tweet["id"]] = {
                "file": filelist[_file_i],
                "index": _tweet_i
            }
    data_dir = "/shared/nas/data/m1/chihan3/data/MIPsProRussiaTwitterAnalysis"
    id_list_file = os.path.join(args.data_dir, "id_list.json")
    with open(id_list_file, "w") as f:
        json.dump(id_list, f)

    tweet_ids = list(id_list.keys())
    my_themes = [_id for _id in tweet_ids if
                 id_list[_id]["file"].startswith("26") or
                 id_list[_id]["file"].startswith("27")
                 ]

    def search(index, silent=False, current_level=10):
        if isinstance(index, int):
            index = tweet_ids[index]
        if index not in id_list:
            return 0
        filename = os.path.join(data_dir,
                                id_list[index]["file"])
        with open(filename, "r") as f:
            _tweet = json.loads(f.readlines()[id_list[index]["index"]])
        if not silent:
            with open(filename.replace("tweets", "users"), "r") as f:
                _users = list(map(json.loads, f.readlines()))
                _users = {_user["id"]: _user for _user in _users}
                _user = _users[_tweet["author_id"]]
        depth = 0
        ref_list = _tweet.get("referenced_tweets", [])
        for ref in ref_list:
            depth = max(depth, search(ref["id"], silent, current_level-1))
            if len(ref_list) > 1 and not silent:
                print("-----" * current_level, current_level, ref["type"])
        if not silent:
            print({
                "id": index,
                "text": _tweet["text"],
                "author_id": _user["id"],
                "author_name": _user["name"],
                "author_username": _user["username"],
                # "author_description": _user["set_description"]
                # "verified": _user["verified"]
            })
        return depth + 1

    depth_list = []
    for _id in tqdm(my_themes):
        depth_list.append(search(_id, silent=True))
    depth_list = np.array(depth_list)

    embed()


if __name__ == "__main__":
    args = parse_args()
    main(args)
