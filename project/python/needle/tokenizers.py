"""
Module for tokenizers.
"""
import string
import re
from collections import Counter
from typing import List
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """
    Base class for tokenizers.
    """
    def __init__(self):
        self.unk_index = 0
        self.pad_index = 1
        self.sos_index = 2
        self.eos_index = 3
        self.default_tokens = {
            '<UNK>': self.unk_index, '<PAD>': self.pad_index,
            '<SOS>': self.sos_index, '<EOS>': self.eos_index
        }
        self.vocab_ = None
        self.reverse_vocab_ = None

    def __len__(self):
        return len(self.vocab_) if self.vocab_ else 0

    def _add_default_tokens(self):
        self.vocab_ = {**self.default_tokens, **self.vocab_}

    def _fill_reverse_vocab(self):
        self.reverse_vocab_ = {idx: token for token, idx in self.vocab_.items()}

    def idx2token(self, item: int) -> str:
        return self.reverse_vocab_.get(item)

    @abstractmethod
    def fit(self, data: List[str]):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, item: str) -> int:
        return self.vocab_.get(item, self.unk_index)


class EngLemmaTokenizer(Tokenizer):
    """
    Tokenizer for English language:
     - removes punctuation
     - applies lemmatization with `spacy` (+ lowering)
     - adds start-of-sequence and end-of-sequence tokens
     - performs truncation and padding
    """
    def __init__(self, max_vocab_size: int = 1_000, nlp_batch_size: int = 200):
        super().__init__()

        self.punkt = re.compile(f'[{string.punctuation}]')
        self.max_vocab_size = max_vocab_size

        import spacy
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger', 'attribute_ruler'])
        self.nlp_batch_size = nlp_batch_size

    def _preprocess(self, texts: List[str]) -> List[List[str]]:
        texts = [re.sub(' +', ' ', re.sub(self.punkt, ' ', text)) for text in texts]
        return [[token.lemma_ for token in text if token.lemma_]
                for text in self.nlp.pipe(texts, batch_size=self.nlp_batch_size)]

    def fit(self, data: List[str]):
        tokens = [item for sublist in self._preprocess(data) for item in sublist]
        tokens = [token for token, _ in Counter(tokens).most_common(self.max_vocab_size)]
        self.vocab_ = {token: idx for idx, token in enumerate(tokens, len(self.default_tokens))}
        self._add_default_tokens()
        self._fill_reverse_vocab()
        return self

    def __call__(self, text: str, max_length: int = None) -> List[int]:
        tokens = self._preprocess([text])[0]
        indices = [self.__getitem__(token) for token in tokens]
        indices = [self.sos_index] + indices + [self.eos_index]
        if max_length:
            indices = indices[:max_length]
            if indices[-1] != self.eos_index:
                indices[-1] = self.eos_index
            indices += [self.pad_index] * (max_length - len(indices))
        return indices
