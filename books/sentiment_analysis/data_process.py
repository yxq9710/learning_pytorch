# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 18:13
# @Author  : yxq
# @File    : data_process.py


def load_sentence_polarity():
    from nltk.corpus import sentence_polarity
    from vocab import Vocab
    vocab = Vocab(sentence_polarity.sents())
    train_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in sentence_polarity.sents(categories='pos')[:4000]] + \
                 [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in sentence_polarity.sents(categories='neg')[:4000]]
    test_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in sentence_polarity.sents(categories='pos')[4000:]] + \
                [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in sentence_polarity.sents(categories='neg')[4000:]]
    return train_data, test_data, vocab


def main():
    train_data, test_data, vocab = load_sentence_polarity()
    print("yes")


if __name__ == '__main__':
    main()

