import torch
import torch.optim as optim
from torch import nn
from torch.utils import data as Data

from config.config import load_config
from datasets.dataset import Dataset
from datasets.util import load_data
import logging
from models.model.transformer import Transformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # default print to console
)

if __name__ == '__main__':
    logging.info('Loading config...')
    config = load_config("config/config.yaml")

    enc_inputs, dec_inputs, tgt_outputs, src_vocab_size, tgt_vocab_size, _ = load_data(config.sentences_path, config.src_vocab_path, config.tgt_vocab_path)

    loader = Data.DataLoader(Dataset(enc_inputs, dec_inputs, tgt_outputs), config.batch_size, True)

    model = Transformer(d_model=config.d_model, n_head=config.n_head, d_key=config.d_key,
                        d_value=config.d_value, d_feedforward=config.d_feedforward, d_embed=src_vocab_size,
                        max_len=config.max_len, num_encoder_layers=config.num_encoder_layers, num_decoder_layers=config.num_decoder_layers,
                        tgt_vocab_size=tgt_vocab_size, dropout=config.dropout)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-8)

    logging.info('Start training...')
    for epoch in range(config.epoch):
        for enc_inputs, dec_inputs, tgt_outputs in loader:
            logging.info('Start training epoch %04d...', epoch + 1)
            enc_inputs, dec_inputs, tgt_outputs = enc_inputs, dec_inputs, tgt_outputs
            outputs = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, tgt_outputs.view(-1))
            logging.info('Finish training epoch %04d, loss %.6f ...', epoch + 1, loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    logging.info('Training finished! Start to save model...')
    torch.save(model, 'model.pth')
    logging.info('Model saved!')


