# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 13:17
# @Author  : yxq
# @File    : data_process.py

from vocab import Vocab
import torch
from torch.nn.utils.rnn import pad_sequence


def load_treebank():
    from nltk.corpus import treebank
    sents, postage = zip(*(zip(*sent) for sent in treebank.tagged_sents()))

    vocab = Vocab.build(sents, reversed_tokens=['<pad>'])
    tag_vocab = Vocab.build(postage)

    train_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags))
                  for sentence, tags in zip(sents[:3000], postage[:3000])]
    test_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags))
                 for sentence, tags in zip(sents[3000:], postage[3000:])]
    # print(vocab['<pad>'])
    return train_data, test_data, vocab, tag_vocab


def collect_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = [torch.tensor(ex[1]) for ex in examples]

    inputs = pad_sequence(inputs, batch_first=True, padding_value=1)  # vocab['<pad>'] = 1
    targets = pad_sequence(targets, batch_first=True, padding_value=1)
    return inputs, targets, lengths, inputs != 1


def length_to_mask(lengths, device=None):
    max_len = torch.max(lengths)
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    if device is not None:
        mask = mask.to(device)
    return mask


def main():
    load_treebank()


if __name__ == '__main__':
    main()
