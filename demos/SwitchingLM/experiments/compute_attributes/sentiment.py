# import gensim
import json
# from STOPWORDS import stopwords
# from sentence_transformers import SentenceTransformer, util
import time
# import os
# import sys
import stanza
import stanza.models.classifiers.cnn_classifier as cnn_classifier


chi_nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment',
                          tokenize_no_ssplit=True)


def compute_sentiment(response):
    doc = chi_nlp(response)
    return doc.sentences[0].sentiment


def main():
    nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
    sentiment_processor = nlp.processors['sentiment']
    model = sentiment_processor._model
    print("finish loading model")
    for split in ['train', 'test', 'valid']:
        print(split)
        indx = 0
        start_time = time.time()
        f_w = open('sentiment/' + split + '.sentiment', "w")
        f_r = open(split + '.target', 'r')
        line = f_r.readline()
        while line:
            if indx % 10000 == 0:
                end_time = time.time()
                print(indx, end_time - start_time)
                start_time = time.time()
            indx += 1

        line = json.loads(line)
        response = line["response"]

        response_label = cnn_classifier.label_text(model, [response])

        f_w.write(response_label.strip() + "\n")
        line = f_r.readline()

    print("finished")


if __name__ == "__main__":
    main()
