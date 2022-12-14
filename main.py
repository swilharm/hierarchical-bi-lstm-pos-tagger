import argparse
import time
import random
from collections import Counter

import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm

from conllu import parse
from polyglot.mapping import Embedding

from models import POS_Tagger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Main:
    """Main class representing the entire process for one language and configuration"""

    def __init__(self, language, model_type=('w', 'c'), polyglot=False, freqbin=False):
        with open(f"languages/{language}/{language}-ud-train.conllu") as file:
            self.train_data = parse(file.read())
            random.seed(0)
            self.train_data = random.sample(self.train_data, len(self.train_data))[:5000]
        with open(f"languages/{language}/{language}-ud-dev.conllu") as file:
            self.dev_data = parse(file.read())
        with open(f"languages/{language}/{language}-ud-test.conllu") as file:
            self.test_data = parse(file.read())
        self.embeds = Embedding.from_glove(f"polyglot/{language}.polyglot.txt")

        self.n_epochs = 20
        self.report_every = 1
        self.learning_rate = 0.1
        self.noise = 0.2

        self.model_type = model_type
        self.polyglot = polyglot
        self.freqbin = freqbin

    def build_indexes(self):
        """
        Constructs index dictionaries for words, characters, bytes and labels.
        Also builds the freqbin dictionary and constructs the word embeddings.
        """
        self.w2i = {}
        self.c2i = {}
        self.b2i = {}
        self.l2i = {}
        self.w2i["_UNK"] = 0
        self.c2i["_UNK"] = 0
        self.c2i["<w>"] = 1
        self.c2i["</w>"] = 2
        self.b2i["_UNK"] = 0
        self.b2i["<w>"] = 1
        self.b2i["</w>"] = 2
        self.l2i = {"ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CONJ": 4, "DET": 5,
                    "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10,
                    "PROPN": 11, "PUNCT": 12, "SCONJ": 13, "SYM": 14, "VERB": 15,
                    "X": 16}
        self.freqbin_dict = {}
        tokens = []
        for sentence in self.train_data:
            for token in sentence:
                if type(token['id']) != int:
                    continue
                word = token['form'].lower()
                if word not in self.w2i:
                    self.w2i[word] = len(self.w2i)
                for character in word:
                    if character not in self.c2i:
                        self.c2i[character] = len(self.c2i)
                    for byte in character.encode("utf-8"):
                        if byte not in self.b2i:
                            self.b2i[byte] = len(self.b2i)
                tokens.append(token['form'].lower())
        for word, frequency in Counter(tokens).items():
            self.freqbin_dict[word] = int(np.log(frequency))

        if self.polyglot:
            torch.manual_seed(0)
            self.embedding_matrix = torch.FloatTensor(size=(len(self.w2i), 64))
            for word, index in self.w2i.items():
                embedding = self.embeds.get(word.lower())
                if self.polyglot and embedding is not None:
                    self.embedding_matrix[index] = torch.FloatTensor(embedding)
                else:
                    self.embedding_matrix[index] = torch.rand((1, 64))
        else:
            self.embedding_matrix = torch.rand((len(self.w2i), 128))

    def tensorize_data(self, sentence):
        """
        Turns sentence into tensor format
        :return: tensor of tokens, lists of characters and bytes per word and the expected labels
        """
        tokens_list = []
        char_lists = []
        byte_lists = []
        pos_tag_list = []
        freq_list = []
        for token in sentence:
            if type(token['id']) != int:
                continue
            word = token['form'].lower()
            tokens_list.append(self.w2i[word] if word in self.w2i else 0)
            char_list = [self.c2i['<w>']]
            byte_list = [self.b2i['<w>']]
            for char in word:
                char_list.append(self.c2i[char] if char in self.c2i else 0)
                for byte in char.encode('utf-8'):
                    byte_list.append(self.b2i[byte] if byte in self.b2i else 0)
            char_list.append(self.c2i['</w>'])
            byte_list.append(self.b2i['</w>'])
            char_lists.append(char_list)
            byte_lists.append(byte_list)
            pos_tag_list.append(self.l2i[token['upos']])
            freq_list.append(self.freqbin_dict[word] if word in self.freqbin_dict else 0)
        tokens = torch.LongTensor(tokens_list).to(device)
        pos_gold = torch.LongTensor(pos_tag_list).to(device)
        freq_gold = torch.LongTensor(freq_list).to(device)
        return tokens, char_lists, byte_lists, pos_gold, freq_gold

    def eval(self, data):
        """
        Evaluates the model on the data
        :param data: The data to evaluate on
        :return: Accuracy of model on data
        """
        self.model.eval()
        with torch.no_grad():
            accuracy = 0
            n_tokens = 0
            for sentence in tqdm(data, desc="Evaluation"):
                tokens, char_lists, byte_lists, golden, _ = self.tensorize_data(sentence)
                pred, _ = self.model(tokens, char_lists, byte_lists)
                pred_label = torch.argmax(pred, dim=1)
                accuracy += torch.sum(pred_label == golden)
                n_tokens += len(tokens)
        return accuracy / n_tokens

    def train(self):
        """
        Trains the model with the training data, using the hyperparameters defined in the constructor.
        """

        self.model = POS_Tagger(self.model_type, self.polyglot, self.freqbin, self.embedding_matrix,
                                len(self.c2i), len(self.b2i), max(self.freqbin_dict.values()) + 1, self.noise
                                ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.n_epochs):
            total_loss = 0
            self.model.train()
            for sentence in tqdm(self.train_data, desc="Training  "):
                optimizer.zero_grad()
                tokens, char_lists, byte_lists, pos_gold, freq_gold = self.tensorize_data(sentence)
                pos_pred, freq_pred = self.model(tokens, char_lists, byte_lists)
                loss = criterion(pos_pred, pos_gold)
                if self.freqbin:
                    loss += criterion(freq_pred[:-1], freq_gold[1:])
                total_loss += loss
                loss.backward()
                optimizer.step()

            # Testing
            if ((epoch + 1) % self.report_every) == 0:
                train_accuracy = self.eval(self.train_data)
                dev_accuracy = self.eval(self.dev_data)
                loss = total_loss / len(self.train_data)
                print(f"epoch: {epoch}, loss: {loss:.4f}, train acc: {train_accuracy:.4f}, dev acc: {dev_accuracy:.4f}")

    def load_model(self, path):
        """
        Loads model from disk
        """
        self.model = POS_Tagger(self.model_type, self.polyglot, self.freqbin, self.embedding_matrix,
                                len(self.c2i), len(self.b2i), max(self.freqbin_dict.values()) + 1, self.noise
                                ).to(device)
        self.model.load_state_dict(torch.load(path))

    def test(self):
        """
        Evaluates model on test data
        :return: Accuracy
        """
        test_accuracy = self.eval(self.test_data)
        print(f"\nTest accuracy: {test_accuracy}")
        return test_accuracy


models = [
    {'model_type': ('w',), 'polyglot': False, 'freqbin': False},
    {'model_type': ('c',), 'polyglot': False, 'freqbin': False},
    {'model_type': ('c', 'b'), 'polyglot': False, 'freqbin': False},
    {'model_type': ('w', 'c'), 'polyglot': False, 'freqbin': False},
    {'model_type': ('w', 'c'), 'polyglot': True, 'freqbin': False},
    {'model_type': ('w', 'c'), 'polyglot': True, 'freqbin': True},
]
languages = ['ar', 'bg', 'cs', 'da', 'de', 'en', 'es', 'eu', 'fa', 'fi', 'fr', 'he',
             'hi', 'hr', 'id', 'it', 'nl', 'no', 'pl', 'pt', 'sl', 'sv']


def model_name(model):
    """Stringifies the model"""
    out = '+'.join(model['model_type'])
    if model['polyglot']:
        out += '_p'
    if model['freqbin']:
        out += '_f'
    return out


def train():
    """
    Trains a model for each language and combination of submodels
    Stores results in training_results.csv
    """
    with open("training_results.csv", 'w') as file:
        file.write(f"Language, Model, Accuracy, Time\n")
    with open("training_results.csv", 'a', 1) as file:
        for language in languages:
            print(f"Training language {language}")
            pos_tagger = Main(language, **model)
            for model in models:
                print(f"\twith model {model_name(model)}")
                pos_tagger.model_type = model['model_type']
                pos_tagger.polyglot = model['polyglot']
                pos_tagger.freqbin = model['freqbin']
                pos_tagger.build_indexes()
                start = time.time()
                pos_tagger.train()
                end = time.time()
                file.write(f"{language}, {model_name(model)}, {pos_tagger.test()}, {end - start}\n")
                torch.save(pos_tagger.model.state_dict(), f"models/{language}_{model_name(model)}.pt")


def evaluate():
    """
     Loads the trained models from disk and evaluates them on test data
     Stores results in inference_results.csv
    """
    with open("inference_results.csv", 'w') as file:
        file.write(f"Language, Model, Accuracy, Time\n")
    with open("inference_results.csv", 'a', 1) as file:
        for language in languages:
            print(f"Evaluating language {language}")
            pos_tagger = Main(language, **model)
            for model in models:
                print(f"\twith model {model_name(model)}")
                pos_tagger.model_type = model['model_type']
                pos_tagger.polyglot = model['polyglot']
                pos_tagger.freqbin = model['freqbin']
                pos_tagger.build_indexes()
                pos_tagger.load_model(f"models/{model_name(model)}_{language}.pt")
                start = time.time()
                accuracy = pos_tagger.test()
                end = time.time()
                file.write(f"{language}, {model_name(model)}, {accuracy}, {end - start}\n")


def parse_arguments():
    """
    Parses program parameters
    """
    parser = argparse.ArgumentParser(
        description="Pytorch implementation of paper \"Multilingual Part-of-Speech Tagging with "
                    "Bidirectional Long Short-Term Memory Models and Auxiliary Loss\" (Plank et al., 2016)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_const', const=True, default=False,
                       help='Use this option to train all the models', required=False)
    group.add_argument('--eval', action='store_const', const=True, default=False,
                       help='Use this option to evaluate all the models', required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    if args.train:
        train()
    else:
        evaluate()
