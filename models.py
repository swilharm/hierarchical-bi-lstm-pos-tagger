import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Character or Byte-based encoder, applies embedding and runs the input through a BiLSTM. Inner model
    """

    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=100)
        self.bilstm = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, bidirectional=True)

    def forward(self, chars: torch.tensor):  # chars = (N_CHARS,)
        embedded = self.embedding(chars)
        _, (final_hidden, _) = self.bilstm(embedded.view(len(chars), 1, 100))
        return final_hidden.view(-1)  # (2*HIDDEN,)


class POS_Tagger(nn.Module):
    """
    Outer model embedding the word and/or using the character/byte encoder.
    Calculates POS tag and freqbin label of the next word
    Applied Gaussian noise to the embedded inputs
    """

    def __init__(self, model_type, use_polyglot, use_freqbin, embedding_matrix, c_vocab_size, b_vocab_size, freq_max, noise):
        super(POS_Tagger, self).__init__()
        torch.manual_seed(0)
        self.model_type = model_type
        self.use_polyglot = use_polyglot
        self.use_freqbin = use_freqbin
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix).requires_grad_(True)
        self.characterbased = Encoder(c_vocab_size).to(device)
        self.bytebased = Encoder(b_vocab_size).to(device)
        self.input_size = 0
        if 'w' in model_type:
            self.input_size += 64 if use_polyglot else 128
        if 'c' in model_type:
            self.input_size += 200
        if 'b' in model_type:
            self.input_size += 200
        self.bilstm = nn.LSTM(input_size=self.input_size, hidden_size=100, num_layers=1, bidirectional=True)
        self.pos_tagger = nn.Linear(in_features=200, out_features=17)
        self.freqbin = nn.Linear(in_features=200, out_features=freq_max)
        self.noise = noise

    def forward(self, tokens: torch.tensor, char_lists: list,
                byte_lists: list):  # tokens = (N_TOKENS, 64), char_lists = List[List[int]]
        concatted = torch.zeros((len(tokens), self.input_size), device=device)  # concatted = (N_TOKENS, 264)
        if 'w' in self.model_type:
            embedded_words = self.embedding(tokens)
        for i, (char_list, byte_list) in enumerate(zip(char_lists, byte_lists)):
            embedded = torch.zeros((0,), device=device)
            if 'w' in self.model_type:
                embedded = torch.concat((embedded, embedded_words[i]))
            if 'c' in self.model_type:
                embedded_characters = self.characterbased(torch.tensor(char_list, device=device))
                embedded = torch.concat((embedded, embedded_characters))
            if 'b' in self.model_type:
                embedded_bytes = self.bytebased(torch.tensor(byte_list, device=device))
                embedded = torch.concat((embedded, embedded_bytes))
            concatted[i] = embedded
        if self.training:
            noise = torch.autograd.Variable(concatted.data.new(concatted.size()).normal_(0, self.noise))
            concatted = concatted + noise
        bilstm_out, _ = self.bilstm(concatted.view(len(tokens), 1, self.input_size))
        pos_tags = self.pos_tagger(bilstm_out.view(len(tokens), 200))
        if self.use_freqbin:
            freq = self.freqbin(bilstm_out.view(len(tokens), 200))
            return pos_tags, freq
        else:
            return pos_tags, None
