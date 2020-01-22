import numpy as np
import pandas as pd
import os
import pickle
from utils import config
import codecs


class BertInputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, actual_input_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.actual_input_id = actual_input_id


# added by Soroush
def save_obj(obj, name):
    with open('./data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# added by Soroush
def load_obj(name):
    with open('./data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def to_lower_case(tweet):
    return tweet.lower()


def load_dataset(path):
    if not os.path.exists(path + config.TRAIN_FILE):
        raise ValueError('DataSet path does not exist ')
        return

    if config.load_dataset_from_pickle == True:

        data = load_obj('train')
        print(len(data))
        test = load_obj('test')
        print(len(test))
        valid = load_obj('valid')
        print(len(valid))

    else:
        data = pd.read_csv(path + config.TRAIN_FILE, sep='\t')

        data['tweet'] = data['tweet'].apply(to_lower_case)
        data['subtask_a'] = data['subtask_a'].map(lambda label: 1 if label == 'OFF' else 0)

        print(len(data))

        test = data.sample(500)
        data = data.drop(test.index)
        valid = data.sample(100)
        data = data.drop(valid.index)

        print(len(data))
        print(len(test))
        print(len(valid))

        save_obj(data, 'train')
        save_obj(test, 'test')
        save_obj(valid, 'valid')

    return data, test, valid


def convert_examples_to_features(pandas, max_seq_length, tokenizer):
    features = []

    for i, r in pandas.iterrows():

        first_tokens = tokenizer.tokenize(r['tweet'])
        if len(first_tokens) > max_seq_length - 2:
            first_tokens = first_tokens[: max_seq_length - 2]
        tokens = ["[CLS]"] + first_tokens + ["[SEP]"]

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        features.append(
            BertInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=r['subtask_a'],
                actual_input_id=r['id']))
    return features
