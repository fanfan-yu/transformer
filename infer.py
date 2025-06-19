import logging
import torch

from torch.utils import data as Data
from config.config import load_config
from datasets.dataset import Dataset
from datasets.util import load_data, read_vocab

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # default print to console
)

def load_model():
    model = torch.load('./model.pth')
    return model

def execute_infer(model, enc_inputs, tgt_len, word2id):
    enc_outputs, enc_self_attns = model.encoder(enc_inputs)
    dec_inputs = torch.zeros(1, tgt_len).type_as(enc_inputs.data)

    # get decoder input
    next_symbol = word2id["<s>"]
    for idx in range(0, tgt_len):
        dec_inputs[0][idx] = next_symbol
        dec_outputs, _, _ = model.decoder(dec_inputs, enc_inputs, enc_outputs)
        projected = model.linear(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_symbol = prob.data[idx].item()

    predict = model(enc_inputs[0].view(1, -1), dec_inputs)
    return predict.data.max(1, keepdim=True)[1]

if __name__ == '__main__':
    logging.info('Loading config...')
    modelConfig = load_config("config/config.yaml")

    # load vocab: id2word and word2id
    word2id = read_vocab(modelConfig.tgt_vocab_path)
    idx2word = {word2id[key]: key for key in word2id}

    logging.info('Loading data...')
    enc_inputs, dec_inputs, tgt_outputs, _, _, tgt_len = load_data(modelConfig.sentences_path, modelConfig.src_vocab_path, modelConfig.tgt_vocab_path)
    load = Data.DataLoader(Dataset(enc_inputs, dec_inputs, tgt_outputs), modelConfig.batch_size, True)


    logging.info('Execute inference...')
    model = load_model()
    predict = execute_infer(model, enc_inputs[0].view(1, -1), tgt_len, word2id)

    print([idx2word[n.item()] for n in predict.squeeze()])



