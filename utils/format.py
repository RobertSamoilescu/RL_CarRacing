import os
import json
import numpy
import re
import torch
import torch_rl
import gym

import utils

def get_obss_preprocessor(env_id, obs_space, model_dir):

    # Check if the obs_space is of type Box([X, Y, 3])
    obs_space = {
        "image": obs_space[0].shape,
        "action": obs_space[1],
        "params": obs_space[2]
    }

    def preprocess_obss(obss, device=None):
        return torch_rl.DictList({
            "image": preprocess_images([obs["image"] for obs in obss], device=device),
            "action": preprocess_actions([obs["action"] for obs in obss], device=device),
            "params": preprocess_params([obs["params"] for obs in obss], device=device)
    })

    return obs_space, preprocess_obss

def preprocess_images(images, mean_value=128., device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    images = (images - mean_value) / mean_value
    return torch.tensor(images, device=device, dtype=torch.float)

def preprocess_actions(actions, device=None):
    return torch.tensor(actions, device=device, dtype=torch.float)

def preprocess_params(params, device=None):
    return torch.tensor(params, device=device, dtype=torch.float)

def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)

class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, model_dir, max_size):
        self.path = utils.get_vocab_path(model_dir)
        self.max_size = max_size
        self.vocab = {}
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self):
        utils.create_folders_if_necessary(self.path)
        json.dump(self.vocab, open(self.path, "w"))