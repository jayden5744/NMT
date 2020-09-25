# -*- coding: utf-8 -*-
import os
import json
from source.data_helper import create_or_get_voca


def load_json(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data


class Helper:
    def __init__(self, config_path):
        data = load_json(config_path)
        self.file_path = data['file_path']
        self.basic_parameter = data['basic_parameter']
        self.seq2seq = data['seq2seq']
        self.attention = data['attention']
        self.transformer = data['transformer']
        self.x_train_path = os.path.join(self.file_path['data_path'],
                                         self.file_path['src_train_filename'])   # train input 경로
        self.y_train_path = os.path.join(self.file_path['data_path'],
                                         self.file_path['tar_train_filename'])   # train target 경로
        self.x_val_path = os.path.join(self.file_path['data_path'],
                                       self.file_path['src_val_filename'])  # validation input 경로
        self.y_val_path = os.path.join(self.file_path['data_path'],
                                       self.file_path['tar_val_filename'])  # validation target 경로
        self.label_smoothing = self.basic_parameter['label_smoothing']
        self.early_stopping = self.basic_parameter['early_stopping']
        self.nmt_model = self.basic_parameter['model']
        self.max_sequence_size = self.basic_parameter['max_sequence_size']
        self.ko_voc, self.en_voc = self.get_voca()

    def encoder_parameter(self):
        if self.nmt_model == 'seq2seq':
            param = {
                'embedding_size': self.basic_parameter.vocab_size,
                'embedding_dim': self.seq2seq.embedding_dim,
                'pad_id': self.ko_voc['<pad>'],
                'rnn_dim': self.seq2seq.encoder.encoder_rnn_dim,
                'rnn_bias': True,
                'n_layers': self.seq2seq.encoder.encoder_n_layers,
                'embedding_dropout': self.seq2seq.encoder.encoder_embedding_dropout,
                'rnn_dropout': self.seq2seq.encoder.encoder.encoder_rnn_dropout,
                'dropout': self.seq2seq.encoder.encoder_dropout,
                'residual_used': self.seq2seq.encoder.encoder_residual_used,
                'bidirectional': self.seq2seq.encoder.encoder_bidirectional_used,
                'encoder_output_transformer': self.seq2seq.embedding_dim,
                'encoder_output_transformer_bias': self.seq2seq.encoder.encoder_output_transformer_bias,
                'encoder_hidden_transformer': self.seq2seq.embedding_dim,
                'encoder_hidden_transformer_bias': self.seq2seq.encoder.encoder_hidden_transformer_bias
            }
        elif self.nmt_model == 'attention':
            param = {
                'embedding_size': self.basic_parameter.vocab_size,
                'embedding_dim': self.attention.embedding_dim,
                'pad_id': self.ko_voc['<pad>'],
                'rnn_dim': self.attention.encoder.encoder_rnn_dim,
                'rnn_bias': True,
                'n_layers': self.attention.encoder.encoder_n_layers,
                'embedding_dropout': self.attention.encoder.encoder_embedding_dropout,
                'rnn_dropout': self.attention.encoder.encoder_rnn_dropout,
                'dropout': self.attention.encoder.encoder_dropout,
                'residual_used': self.attention.encoder.encoder_residual_used,
                'bidirectional': self.attention.encoder.encoder_bidirectional_used,
                'encoder_output_transformer': self.attention.embedding_dim,
                'encoder_output_transformer_bias': self.attention.encoder.encoder_output_transformer_bias,
                'encoder_hidden_transformer': self.attention.embedding_dim,
                'encoder_hidden_transformer_bias': self.attention.encoder.encoder_hidden_transformer_bias
            }
        elif self.nmt_model == 'transformer':
            param = {
                'input_dim': self.basic_parameter.vocab_size,
                'hid_dim': self.transformer.encoder.encoder_hidden_dim,
                'n_layers': self.transformer.encoder.encoder_layers,
                'n_heads': self.transformer.encoder.encoder_heads,
                'head_dim': self.transformer.encoder.encoder_head_dim,
                'pf_dim': self.transformer.encoder.encoder_pf_dim,
                'dropout': self.transformer.encoder.encoder_dropout,
                'max_length': self.basic_parameter.max_sequence_size,
                'padding_id': self.ko_voc['<pad>']
            }
        else:
            print("적절하지 않은 model 이름입니다. basic_parameter의 model변수를 확인해주세요")
            raise TypeError
        return param

    def decoder_parameter(self):
        if self.nmt_model == 'seq2seq':
            param = {
                'embedding_size': self.basic_parameter.embedding_size,
                'embedding_dim': self.seq2seq.embedding_dim,
                'pad_id': self.en_voc['<pad>'],
                'rnn_dim': self.seq2seq.decoder.decoder_rnn_dim,
                'rnn_bias': True,
                'n_layers': self.seq2seq.decoder.decoder_n_layers,
                'embedding_dropout': self.seq2seq.decoder.decoder_embedding_dropout,
                'rnn_dropout': self.seq2seq.decoder.decoder_rnn_dropout,
                'dropout': self.seq2seq.decoder.decoder_dropout,
                'residual_used': self.seq2seq.decoder_residual_used
            }
        elif self.nmt_model == 'attention':
            param = {
                'embedding_size': self.basic_parameter.embedding_size,
                'embedding_dim': self.attention.embedding_dim,
                'pad_id': self.en_voc['<pad>'],
                'rnn_dim': self.attention.decoder.decoder_rnn_dim,
                'rnn_bias': True,
                'n_layers': self.attention.decoder.decoder_n_layers,
                'embedding_dropout': self.attention.decoder.decoder_embedding_dropout,
                'rnn_dropout': self.attention.decoder.decoder_rnn_dropout,
                'dropout': self.attention.decoder.decoder_dropout,
                'residual_used': self.attention.decoder.decoder_residual_used,
                'attention_score_func': self.attention.attention_score
            }
        elif self.nmt_model == 'transformer':
            param = {
                'input_dim': self.basic_parameter.vocab_size,
                'hid_dim': self.transformer.decoder.decoder_hidden_dim,
                'n_layers': self.transformer.decoder.decoder_layers,
                'n_heads': self.transformer.decoder.decoder_heads,
                'head_dim': self.transformer.decoder.decoder_head_dim,
                'pf_dim': self.transformer.decoder.decoder_pf_dim,
                'dropout': self.transformer.decoder.decoder_dropout,
                'max_length': self.basic_parameter.max_sequence_size,
                'padding_id': self.en_voc['<pad>']
            }
        else:
            print("적절하지 않은 model 이름입니다. basic_parameter의 model변수를 확인해주세요")
            raise TypeError
        return param

    def get_voca(self):
        try:  # vocabulary 불러오기
            ko_voc, en_voc = create_or_get_voca(save_path=self.file_path.dictionary_path,
                                                ko_vocab_size=self.basic_parameter.encoder_vocab_size,
                                                en_vocab_size=self.basic_parameter.decoder_vocab_size,
                                                region=self.basic_parameter.region)
        except OSError:  # 경로 error 발생 시 각각의 경로를 입력해서 가지고 오기
            ko_voc, en_voc = create_or_get_voca(save_path=self.file_path.dictionary_path,
                                                ko_corpus_path=self.x_train_path,
                                                en_corpus_path=self.y_train_path,
                                                ko_vocab_size=self.basic_parameter.encoder_vocab_size,
                                                en_vocab_size=self.basic_parameter.decoder_vocab_size,
                                                region=self.basic_parameter.region)
        return ko_voc, en_voc


