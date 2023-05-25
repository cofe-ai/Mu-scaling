import numpy as np
from .files import load_text_file_by_line


def stat_params(model, show_detail=False):
    total = []
    for name, param in model.named_parameters():
        p_size = param.nelement()
        total.append(p_size)
        if show_detail:
            print('%-50s %-30s %d' % (name, str(tuple(param.size())), p_size))
    # print("Number of parameter: %.2fM" % (sum(total) / 1e6))
    print("Number of parameter: %.2fM" % (sum(total) / (2 ** 20)))


def stat_dataset_zh(file_path, sentence_sep='[SOS]'):
    raw_texts = load_text_file_by_line(file_path)
    dataset_size = len(raw_texts)
    print('All Data: %d' % dataset_size)

    doc_lengths, sent_lengths, num_sent_in_docs = [], [], []
    for doc in raw_texts:
        sents = doc.split(sentence_sep)
        doc_length = 0
        for sent in sents:
            doc_length += len(sent)
            sent_lengths.append(len(sent))
        doc_lengths.append(doc_length)
        num_sent_in_docs.append(len(sents))

    print('Doc  AVG Length: %d' % np.average(doc_lengths))
    print('Sent AVG Length: %d' % np.average(sent_lengths))
    print('Sent AVG Count : %d' % np.average(num_sent_in_docs))


def stat_dataset_by_tokenized(texts: list):
    num_texts = len(texts)
    print('Docs Num: %d' % num_texts)
    text_lengths = [len(tokens) for tokens in texts]
    print('Length: %d' % np.average(text_lengths))


if __name__ == '__main__':
    pass
