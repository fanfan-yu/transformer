import yaml

class config():
    def __init__(self, configMap):
        self.sentences_path = configMap["path"]["sentences_path"]
        self.src_vocab_path = configMap["path"]["src_vocab_path"]
        self.tgt_vocab_path = configMap["path"]["tgt_vocab_path"]
        self.d_model = configMap["model"]["d_model"]
        self.n_head = configMap["model"]["n_head"]
        self.d_key = configMap["model"]["d_key"]
        self.d_value = configMap["model"]["d_value"]
        self.d_feedforward = configMap["model"]["d_feedforward"]
        self.max_len = configMap["model"]["max_len"]
        self.num_encoder_layers = configMap["model"]["num_encoder_layers"]
        self.num_decoder_layers = configMap["model"]["num_decoder_layers"]
        self.batch_size = configMap["train"]["batch_size"]
        self.epoch = configMap["train"]["epoch"]
        self.learning_rate = configMap["train"]["learning_rate"]
        self.dropout = configMap["train"]["dropout"]

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        configMap = yaml.safe_load(f)
    return config(configMap)