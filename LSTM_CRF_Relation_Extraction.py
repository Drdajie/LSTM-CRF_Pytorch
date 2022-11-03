import torch
import torch.optim as optim
import torch.nn as nn

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def prepare_sequence_batch(data, word_to_ix, tag_to_ix):
    seqs, tags = [d[0] for d in data], [d[1] for d in data]
    max_len = max([len(seq) for seq in seqs])

    mask = []
    lens = [len(seq) for seq in seqs]
    for bi in range(len(data)):
        b_mask = torch.zeros(max_len)
        b_mask[:lens[bi]] = 1
        mask.append(b_mask.unsqueeze(0))
    mask = torch.cat(mask, dim=0)

    seqs_pad, tags_pad = [], []
    for seq, tag_table in zip(seqs, tags):
        seq_pad = seq + ['<PAD>'] * (max_len-len(seq))
        tag_pad = []
        tag_len = len(tag_table)
        for tag_i in range(max_len):
            if tag_i < tag_len:
                tag_pad.append(tag_table[tag_i] + ['<PAD>'] * (max_len-tag_len))
            else:
                tag_pad.append(['<PAD>'] * max_len)
        seqs_pad.append(seq_pad)
        tags_pad.append(tag_pad)
    idxs_pad = torch.tensor([[word_to_ix[w] for w in seq] for seq in seqs_pad], dtype=torch.long)
    tags_pad = torch.tensor([[[tag_to_ix[t] for t in tag] for tag in tag_table] for tag_table in tags_pad], dtype=torch.long)

    return idxs_pad, tags_pad, mask

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF_MODIFY_PARALLEL(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF_MODIFY_PARALLEL, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
    
    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2), 
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features_parallel(self, sentences):
        if len(sentences.shape) > 1:    self.batch_size = sentences.shape[0]
        else:   self.batch_size = 1

        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentences).view(self.batch_size, -1, self.embedding_dim)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        lstm_out_a = lstm_out.unsqueeze(1).expand(lstm_out.shape[0], lstm_out.shape[1],
                                    lstm_out.shape[1], lstm_out.shape[2])
        lstm_out_b = lstm_out_a.transpose(1, 2)

        lstm_out = torch.cat((lstm_out_a, lstm_out_b), -1)

        if self.batch_size == 1:    lstm_out = lstm_out.squeeze(0)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg_new_parallel(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([feats.shape[0], feats.shape[1], self.tagset_size], -10000.)
        # START_TAG has all of the score.
        init_alphas[:, :, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_vars = init_alphas
        for feat_index in range(feats.shape[1]):
            # batch * tagset_size * tagset_size
            pre = torch.unsqueeze(forward_vars, dim=2)
            emit = torch.unsqueeze(feats[:, :, feat_index, :], 2).transpose(2, 3)
            tmp_vars = pre + emit + self.transitions
            forward_vars = torch.logsumexp(tmp_vars, dim=3)

        terminal_var = forward_vars + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = torch.logsumexp(terminal_var, dim=2)
        return alpha

    def _viterbi_decode(self, feats):
        backpointers = []

        max_len = feats.shape[1]

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((max_len, 1, self.tagset_size), -10000.)
        init_vvars[:, 0, self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for i in range(max_len):
            tmp_matrix = forward_var + self.transitions
            best_tag_id = torch.argmax(tmp_matrix, dim=2)
            
            forward_var = torch.cat([tmp_matrix[i, range(self.tagset_size), best_tag_id[i]] \
                             for i in range(max_len)], dim = 0).view(max_len, 1, -1)
            forward_var = (forward_var + feats[:, i, :].unsqueeze(1)).view(max_len, 1, -1)
            backpointers.append(best_tag_id)


        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var = terminal_var.squeeze(1)
        best_tag_id = torch.argmax(terminal_var, dim = 1)
        path_score = terminal_var[range(max_len), best_tag_id]
        print(path_score.shape)

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        print(best_path)
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[range(max_len), best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we don't want to returen that to the caller)
        start = best_path.pop()
        for tmp_start in start:
            assert tmp_start == self.tag_to_ix[START_TAG]    # Sanity check
        best_path.reverse()
        tmp = []
        for xs in best_path:
            tmp.append(xs.numpy().tolist())
        best_path = torch.tensor(tmp)
        return path_score, best_path

    def forward(self, sentence):  
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features_parallel(sentence)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        tag_seq = tag_seq.t()
        return score, tag_seq

    def _score_sentence_parallel(self, feats, tags):
        # Gives the score of provided tag sequences
        #feats = feats.transpose(0,1)
        score = torch.zeros(tags.shape[0], tags.shape[1])#.to('cuda')
        tags = torch.cat([torch.full([tags.shape[0], tags.shape[1], 1], self.tag_to_ix[START_TAG], dtype=torch.long), tags],dim=2)
        feats_index_a, feats_index_b = [[ix] * feats.shape[1] for ix in range(feats.shape[0])], \
            [[ix for ix in range(feats.shape[1])] for tmp in range(feats.shape[0])]

        for i in range(feats.shape[2]):
            feat=feats[:, :, i, :]
            score = score + \
                    self.transitions[tags[:, :,i + 1], tags[:, :,i]] + feat[feats_index_a, feats_index_b, tags[:, :,i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:, :,-1]]
        return score

    def neg_log_likelihood_parallel(self, sentences, tags, mask):
        feats = self._get_lstm_features_parallel(sentences)

        forward_score = self._forward_alg_new_parallel(feats)
        gold_score = self._score_sentence_parallel(feats, tags)

        score = forward_score - gold_score
        score = score * mask
        lens = torch.sum(mask, dim = 1)
        score = torch.sum(score, dim=1)/lens
        return torch.sum(score)/tags.shape[0]



START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"
EMBEDDING_DIM = 300
HIDDEN_DIM = 256


# Make up some training data

training_data = [("some others give the new UMUC 5 stars - don't believe them".split(),
    ["h h o o o o o o o o o o".upper().split(), "o h o o o o o o o o o o".upper().split(), 
    "o o o o o o o o o o o o".upper().split(), "o o o t t t o o o o o o".upper().split(),
    "o o o t t t o o o o o o".upper().split(), "o o o t t t o o o o o o".upper().split(),
    "o o o o o o o o o o o o".upper().split(), "o o o o o o o o o o o o".upper().split(),
    "o o o o o o o o o o o o".upper().split(), "o o o o o o o o o o o o".upper().split(), 
    "o o o o o o o o o o o o".upper().split(), "o o o o o o o o o o o o".upper().split()]),
    ("georgia tech is a university in georgia".split(),
    ["h h o o o o o".upper().split(), "h h o o o o o".upper().split(),
    "o o o o o o o".upper().split(), "o o o t t o o".upper().split(),
    "o o o o t o o".upper().split(), "o o o o o o o".upper().split(),
    "o o o o o o o".upper().split()])
]


word_to_ix = {}
word_to_ix['<PAD>'] = 0
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"H": 0, "E": 1, "T": 2, "O":3, START_TAG: 4, STOP_TAG: 5, PAD_TAG: 6}

model = BiLSTM_CRF_MODIFY_PARALLEL(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.3, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([[tag_to_ix[t] for t in tag_table] for tag_table in training_data[0][1]], \
            dtype=torch.long)
    precheck_sent = precheck_sent.unsqueeze(0)
    # print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(300):
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad()
    # Step 2. Get our batch inputs ready for the network, that is,
    # turn them into Tensors of word indices.
    # If training_data can't be included in one batch, you need to sample
    # them to build a batch.
    sentence_in_pad, targets_pad, mask = prepare_sequence_batch(training_data, word_to_ix, tag_to_ix)
    # Step 3. Run our forward pass.
    loss = model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad, mask)
    # # Step 4. Compute the loss, gradients, and update the parameters by
    # # calling optimizer.step()
    # print(loss)
    loss.backward()
    optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[1][0], word_to_ix)
    res = model(precheck_sent.unsqueeze(0))
    sentence_in_pad, targets_pad, mask = prepare_sequence_batch(training_data, word_to_ix, tag_to_ix)
    # res = model(sentence_in_pad)
    print(targets_pad[1])
    print("res0：", res[0])
    print(res[1])
    # We got it!