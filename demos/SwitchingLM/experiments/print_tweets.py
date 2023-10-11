import json
from pprint import pprint


def print_tweets(filename):
    f = open(filename)
    tweets = list(map(json.loads, f.readlines()))
    pprint([_t['text'] for _t in tweets], width=1000)
