import torch
import logging

# pre process loaded data
def load_data(sentences_path, src_vocab_path, tgt_vocab_path):
    # load data from directory
    logging.info('Loading data from directory')
    sentences = read_sentences_file(sentences_path)

    # load vocab
    logging.info('Loading vocab from directory')
    src_vocab = read_vocab(src_vocab_path)
    tgt_vocab = read_vocab(tgt_vocab_path)

    enc_inputs, dec_inputs, dec_outputs = [], [], []

    # 'i' is the index of sentence, 'j' is the index of word
    for i in range(len(sentences)):
        enc_input = [[src_vocab[j] for j in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[j] for j in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[j] for j in sentences[i][2].split()]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    tgt_len = len(sentences[0][1].split(" "))
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs), len(src_vocab), len(tgt_vocab), tgt_len

# read sentences from dataset directory
def read_sentences_file(sentences_path):
    sentences = []
    with open(sentences_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) == 0:
        raise ValueError("data file is empty")

    # check if the number of lines is even
    if len(lines) % 2 != 0:
        raise ValueError("data file should have even number of lines")

    for i in range(0, len(lines), 2):
        src = lines[i]
        tgt = lines[i + 1]
        # For decoder input, add <s> at the beginning. For decoder output, add </s> at the end
        sentences.append([src, '<s> '+tgt, tgt+' </s>'])

    return sentences

def read_vocab(file_path):
    vocab = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    word = ' '.join(parts[:-1])  #
                    word_id = int(parts[-1])
                    vocab[word] = word_id
                else:
                    logging.warning(f"Skipping invalid line: {line}")
    return vocab
