import random
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StackLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bias=True, dropout=0, residual=True):
        super().__init__()
        self.input_size = input_size            # embedding_dim
        self.hidden_size = hidden_size          # rnn_dim
        self.n_layers = n_layers                # n_layers
        self.layers = nn.ModuleList()
        self.residual = residual                # 잔차연결
        self.dropout = nn.Dropout(p=dropout)    # dropout

        for i in range(n_layers):               # LSTM 층 쌓기
            self.layers.append(
                nn.LSTMCell(input_size, hidden_size, bias=bias)
            )
            input_size = hidden_size

    def forward(self, inputs, hidden):
        # input => [batch_size, rnn_dim]
        # hidden => (h_state, c_state)
        # h_state, c_state = [n_layers, batch_size, hidden_size]
        h_state, c_state = hidden  # 이전 hidden, cell 상태 받아오기

        next_h_state, next_c_state = [], []

        for i, layer in enumerate(self.layers):  # 각 층 layer와 idx
            hi = h_state[i].squeeze(dim=0)
            ci = c_state[i].squeeze(dim=0)
            # squeeze :  차원의 원소가 1인 차원을 모두 없애줌, dim=n : n번째 차원만 1이면 없애줌

            if hi.dim() == 1 and ci.dim() == 1:  # hidden, cell layer의 차원이 1이면
                hi = h_state[i]
                ci = c_state[i]

            next_hi, next_ci = layer(inputs, (hi, ci))
            output = next_hi

            if i + 1 < self.n_layers:  # rnn dropout layer
                output = self.dropout(output)
            if self.residual and inputs.size(-1) == output.size(-1):  # 잔차연결
                inputs = output + inputs
            else:
                inputs = output

            next_h_state.append(next_hi)
            next_c_state.append(next_ci)

        next_hidden = (
            torch.stack(next_h_state, dim=0),       # hidden layer concaternate
            torch.stack(next_c_state, dim=0)        # cell layer concaternate
        )
        # input => [batch_size, rnn_dim]
        # next_hidden => (h_state, c_state)
        # h_state, c_state = [n_layers, batch_size, hidden_size]
        return inputs, next_hidden


class Recurrent(nn.Module):
    def __init__(self, cell, reverse=False):
        super().__init__()
        self.cell = cell
        self.reverse = reverse  # reverse : 양방향 시 사용

    def forward(self, inputs, pre_hidden=None, get_attention=False, attention=None, encoder_outputs=None):
        # inputs => [batch_size, sequence_len, embedding_dim]
        # hidden => (h_state, c_state)
        # h_state, c_state = [n_layers, batch_size, hidden_size]
        hidden_size = self.cell.hidden_size
        batch_size = inputs.size()[0]

        if pre_hidden is None:
            n_layers = self.cell.n_layers
            zero = inputs.data.new(1).zero_()
            # hidden 초기화
            h0 = zero.view(1, 1, 1).expand(n_layers, batch_size, hidden_size)
            # Xavier normal 초기화
            nn.init.xavier_normal_(h0)
            # cell 초기화
            c0 = zero.view(1, 1, 1).expand(n_layers, batch_size, hidden_size)
            hidden = (h0, c0)
        else:
            hidden = pre_hidden
        outputs = []
        attentions = []
        inputs_time = inputs.split(1, dim=1)    # => ([batch_size, 1, embedding_dim] * sequence_len)
        if self.reverse:
            inputs_time = list(inputs_time)
            inputs_time.reverse()

        for input_t in inputs_time:             # sequence_len 만큼 반복
            input_t = input_t.squeeze(1)        # => [batch_size, embedding_dim]
            last_hidden_t, hidden = self.cell(input_t, hidden)
            if get_attention:
                output_t, score = attention(encoder_outputs, last_hidden_t)
                attentions.append(score)
            outputs += [last_hidden_t]

        if self.reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, dim=1)
        # outputs => [batch_size, sequence_len, embedding_dim]
        # hidden => (h_state, c_state)
        # h_state, c_state = [n_layers, batch_size, hidden_size]
        if get_attention:
            attentions = torch.stack(attentions, dim=2)
            return outputs, hidden, attentions
        return outputs, hidden


class BiRecurrent(nn.Module):
    def __init__(self, cell, output_transformer, output_transformer_bias,
                 hidden_transformer, hidden_transformer_bias):
        super().__init__()
        hidden_size = cell.hidden_size * 2
        self.forward_rnn = Recurrent(cell, reverse=False)
        self.reverse_rnn = Recurrent(cell, reverse=True)
        self.output_nn = nn.Linear(hidden_size, output_transformer, bias=output_transformer_bias)
        self.hidden_nn = nn.Linear(hidden_size, hidden_transformer, bias=hidden_transformer_bias)
        self.cell_nn = nn.Linear(hidden_size, hidden_transformer, bias=hidden_transformer_bias)

    def forward(self, inputs, hidden=None):
        forward_output, (forward_hidden, forward_cell) = self.forward_rnn(inputs, hidden)
        reverse_output, (reverse_hidden, reverse_cell) = self.reverse_rnn(inputs, hidden)

        output = torch.cat((forward_output, reverse_output), dim=2)
        output = self.output_nn(output)
        hidden = torch.cat((forward_hidden, reverse_hidden), dim=2)
        hidden = self.hidden_nn(hidden)
        cell = torch.cat((forward_cell, reverse_cell), dim=2)
        cell = self.cell_nn(cell)

        return output, (hidden, cell)


class AttentionEncoder(nn.Module):
    def __init__(self, embedding_size, embedding_dim, rnn_dim, rnn_bias, pad_id, n_layers=1, residual_used=True,
                 embedding_dropout=0, rnn_dropout=0, dropout=0, bidirectional=True,
                 encoder_output_transformer=None, encoder_output_transformer_bias=None,
                 encoder_hidden_transformer=None, encoder_hidden_transformer_bias=None):
        super().__init__()
        self.embedding = nn.Embedding(embedding_size, embedding_dim, padding_idx=pad_id)    # Embedding Layer
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)                            # Embedding dropout
        self.dropout = nn.Dropout(p=dropout)
        # rnn cell
        cell = StackLSTMCell(input_size=self.embedding.embedding_dim, hidden_size=rnn_dim, n_layers=n_layers,
                             bias=rnn_bias, dropout=rnn_dropout, residual=residual_used)
        if bidirectional:
            assert encoder_output_transformer and encoder_output_transformer_bias \
                   and encoder_hidden_transformer and encoder_hidden_transformer_bias, 'not input transformer parameter'
            # 가정설정문 : parameter 중 하나라도 안들어오면 해당 메세지 출력
            self.rnn = BiRecurrent(cell, encoder_output_transformer, encoder_output_transformer_bias,
                                   encoder_hidden_transformer, encoder_hidden_transformer_bias)
        else:
            self.rnn = Recurrent(cell)

    def forward(self, enc_input):
        # enc_input => [batch_size, sequence_len]
        embedded = self.embedding_dropout(self.embedding(enc_input))    # input을 embedding시키고 dropout 적용
        embedded = F.relu(embedded)
        # embedded => [batch_size, sequence_len, embedding_dim]
        output, (hidden, cell) = self.rnn(embedded)
        output = self.dropout(output)
        # output => [batch_size, sequence_len, rnn_dim]
        # hidden => [n_layer, batch_size, rnn_dim]
        # cell => [n_layer, batch_size, rnn_dim]
        return output, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_size, score_function):
        super().__init__()
        if score_function not in ['dot', 'general', 'concat']:
            raise NotImplemented('Not implemented {} attention score function '
                                 'you must selected [dot, general, concat]'.format(score_function))

        self.character_distribution = nn.Linear(hidden_size * 2, hidden_size)
        self.score_function = score_function
        if score_function == 'dot':
            pass
        elif score_function == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            raise NotImplementedError

    def forward(self, context, target):
        # context => [batch_size, seq_len, hidden]
        # target => [batch_size, hidden]
        # batch_size, _ = context.size()
        batch_size, seq_len, _ = context.size()
        if self.score_function == 'dot':
            x = target.unsqueeze(-1)
            attention_weight = context.bmm(x).squeeze(-1)

        elif self.score_function == 'general':
            x = self.Wa(target)
            x = x.unsqueeze(-1)
            attention_weight = context.bmm(x).squeeze(-1)       # => [batch_size, seq_len)
        else:
            raise NotImplementedError

        attention_distribution = F.softmax(attention_weight, -1)    # => [batch_size, seq_len]
        context_vector = attention_distribution.unsqueeze(1).bmm(context).squeeze(1)
        # [batch size, 1, seq_len] * [batch_size, seq_len, hidden] = [batch_size, hidden]
        combine = self.character_distribution(torch.cat((context_vector, target), 1))

        return combine, attention_distribution


class AttentionDecoder(nn.Module):
    def __init__(self, embedding_size, embedding_dim, rnn_dim, rnn_bias, pad_id, n_layers=1, embedding_dropout=0,
                 rnn_dropout=0, dropout=0, residual_used=True, attention_score_func='dot'):
        super().__init__()
        self.vocab_size = embedding_size
        self.hidden_size = rnn_dim              # beam search 적용시 사용하는 변수
        self.embedding = nn.Embedding(embedding_size, embedding_dim, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.dropout = nn.Dropout(p=dropout)

        self.attention = Attention(hidden_size=rnn_dim, score_function=attention_score_func)
        cell = StackLSTMCell(input_size=self.embedding.embedding_dim, hidden_size=rnn_dim, n_layers=n_layers,
                             bias=rnn_bias, residual=residual_used, dropout=rnn_dropout)
        self.rnn = Recurrent(cell)                              # 기본 rnn
        self.classifier = nn.Linear(rnn_dim, embedding_size)    # dense

    def forward(self, dec_input, hidden, encoder_outputs, get_attention=False):
        # dec_intput => [batch_size, seq_len]
        # encoder_outputs => [batch_size, seq_len, hidden]
        # hidden[0] => [n_layers, batch_size, hidden]
        embedded = self.embedding_dropout(self.embedding(dec_input))  # [batch_size, seq_len, hidden]
        embedded = F.relu(embedded)
        if get_attention:
            output, hidden, attention_score = self.rnn(inputs=embedded, pre_hidden=hidden, get_attention=get_attention,
                                                       attention=self.attention, encoder_outputs=encoder_outputs)
            output = self.dropout(output)
            output = self.classifier(output)
            # output => [batch_size, sequence_size, embedding_size]
            return output, hidden, attention_score

        else:
            output, hidden = self.rnn(inputs=embedded, hidden=hidden)
            # output => [batch_size, sequence_size, rnn_dim]
            output = self.dropout(output)
            # output => [batch_size, sequence_size, rnn_dim]
            output = self.classifier(output)  # dense 라인 적용
            # output => [batch_size, sequence_size, embedding_size]
            return output, hidden


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, seq_len, get_attention, beam_search=False, k=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seq_len = seq_len
        self.get_attention = get_attention
        self.beam_search = beam_search
        self.k = k

    def forward(self, enc_input, dec_input, teacher_forcing_rate=0.5):
        # enc_input, dec_input => [batch_size, sequence_len]
        # seed_val = 42
        # random.seed(seed_val)
        encoder_output, pre_hidden = self.encoder(enc_input)
        # output => [batch_size, sequence_len, rnn_dim]
        # pre_hidden => (hidden, cell)
        # hidden => [n_layer, batch_size, rnn_dim]
        # cell => [n_layer, batch_size, rnn_dim]

        # teacher forcing ratio check
        if teacher_forcing_rate == 1.0:  # 교사강요 무조건 적용  => 답을 그대로 다음 input에 넣음
            if self.get_attention:
                output, _, attentions = self.decoder(encoder_outputs=encoder_output, dec_input=dec_input,
                                                     hidden=pre_hidden, get_attention=True)
            else:
                output, _ = self.decoder(encoder_outputs=encoder_output, dec_input=dec_input,
                                         hidden=pre_hidden, get_attention=False)
        else:
            outputs = []
            dec_input_i = dec_input[:, 0].unsqueeze(dim=1)
            if self.get_attention:
                attentions = []
                for i in range(1, self.seq_len + 1):
                    output, pre_hidden, attention = self.decoder(encoder_outputs=encoder_output,
                                                                 dec_input=dec_input_i, hidden=pre_hidden,
                                                                 get_attention=True)
                    _, indices = output.max(dim=2)

                    output = output.squeeze(dim=1)
                    attention = attention.squeeze(dim=2)
                    outputs.append(output)
                    attentions.append(attention)

                    if i != self.seq_len:
                        dec_input_i = \
                            dec_input[:, i].unsqueeze(dim=1) if random.random() < teacher_forcing_rate else indices

                output = torch.stack(outputs, dim=1)
                attentions = torch.stack(attentions, dim=2)
            else:
                for i in range(1, self.seq_len + 1):
                    output, pre_hidden = self.decoder(encoder_outputs=encoder_output, dec_input=dec_input_i,
                                                      hidden=pre_hidden,  get_attention=False)
                    _, indices = output.max(dim=2)

                    output = output.squeeze(dim=1)
                    outputs.append(output)

                    if i != self.seq_len:
                        dec_input_i = \
                            dec_input[:, i].unsqueeze(dim=1) if random.random() < teacher_forcing_rate else indices

                output = torch.stack(outputs, dim=1)
        output = F.log_softmax(output, dim=1)
        if self.get_attention:
            return output, attentions
        return output
