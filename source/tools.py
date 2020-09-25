# -*- coding:utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from source.Utils.utils import Helper
from source.data_helper import TrainDataset, TestDataset
from source.Utils.early_stopping import EarlyStopping
from source.Utils.cross_entropy import CrossEntropyLoss
from source.Utils.optimizer import NoamOpt, InverseSqrt

from source.Models.seq2seq import Seq2SeqEncoder, Seq2SeqDecoder, Seq2Seq
from source.Models.seq2seq_attention import AttentionEncoder, AttentionDecoder, Seq2SeqWithAttention
from source.Models.transformer import TransformerEncoder, TransformerDecoder, Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    """
    Xivier Initializer
    """
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def cal_teacher_forcing_ratio(learning_method, total_step):
    """
    Teacher Forcing / Scheduled Sampling 기법
    :param learning_method: 기법 종류 [Teacher_Forcing, Scheduled_Sampling]
    :param total_step: 총 step 수 (epoch * (len(train_loader) + 1)
    :return: 확률이 담긴 list
    """
    if learning_method == 'Teacher_Forcing':
        teacher_forcing_ratios = [1.0 for _ in range(total_step)]  # 교사강요
    elif learning_method == 'Scheduled_Sampling':
        import numpy as np
        teacher_forcing_ratios = np.linspace(0.0, 1.0, num=total_step)[::-1]  # 스케줄 샘플링
        # np.linspace : 시작점과 끝점을 균일하게 toptal_step수 만큼 나눈 점을 생성
    else:
        raise NotImplementedError('learning method must choice [Teacher_Forcing, Scheduled_Sampling]')
    return teacher_forcing_ratios


class Trainer(Helper):  # Train
    def __init__(self, config_path):
        super().__init__(config_path)
        self.train_loader = self.get_train_loader()  # train data loader
        self.val_loader = self.get_val_loader()  # validation data loader           # cross entropy
        self.criterion = CrossEntropyLoss(ignore_index=self.en_voc['<pad>'], smooth_eps=self.label_smoothing,
                                          from_logits=False)
        self.writer = SummaryWriter()  # tensorboard 기록
        self.early_stopping = EarlyStopping(patience=self.early_stopping, verbose=True)
        self.train()  # train 실행

    def train(self):
        start = time.time()  # 모델 시작 시간 기록
        encoder_parameter = self.encoder_parameter()  # encoder parameter
        decoder_parameter = self.decoder_parameter()  # decoder parameter
        model = self.select_model(encoder_parameter, decoder_parameter)
        model = nn.DataParallel(model).cuda()
        model.train()

        print(f'The model has {count_parameters(model):,} trainable parameters')
        model.apply(initialize_weights)

        # optimizer
        if self.nmt_model in ['seq2seq', 'attention']:
            encoder_optimizer = opt.Adam(model.parameters(), lr=self.basic_parameter.learning_rate)  # Encoder Adam Optimizer
            decoder_optimizer = opt.Adam(model.parameters(), lr=self.basic_parameter.learning_rate)  # Decoder Adam Optimizer
        else:
            optimizer = InverseSqrt(self.basic_parameter.vocab_size, warmup_end_lr=self.basic_parameter.learning_rate,
                                    optimizer=opt.Adam(model.parameters(), lr=self.basic_parameter.learning_rate,
                                                       betas=(0.9, 0.98), weight_decay=0.0001))
        epoch_step = len(self.train_loader) + 1  # 전체 데이터 셋 / batch_size
        total_step = self.basic_parameter.epochs * epoch_step  # 총 step 수

        if self.nmt_model == 'seq2seq':
            train_ratios = cal_teacher_forcing_ratio(self.seq2seq.learning_method, total_step)  # train learning method
            val_ratios = cal_teacher_forcing_ratio('Scheduled_Sampling',
                                                   int(total_step / 100) + 1)  # val learning method
        elif self.nmt_model == 'attention':
            train_ratios = cal_teacher_forcing_ratio(self.attention.learning_method, total_step) # train learning method
            val_ratios = cal_teacher_forcing_ratio('Scheduled_Sampling',
                                                   int(total_step / 100) + 1)  # val learning method

        step = 0

        # for epoch in range(self.basic_parameter.epochs):            # 매 epoch 마다






    def select_model(self, encoder_parameter, decoder_parameter):
        """
        Seq2Seq, Attention, Transformer 중 모델 결정
        :param encoder_parameter: encoder의 파라미터
        :param decoder_parameter: decoder의 파라미터
        :return: model
        """
        if self.nmt_model == 'seq2seq':
            encoder = Seq2SeqEncoder(**encoder_parameter)   # encoder 초기화
            decoder = Seq2SeqDecoder(**decoder_parameter)   # decoder 초기화
            model = Seq2Seq(encoder, decoder, self.max_sequence_size)
        elif self.nmt_model == 'attention':
            encoder = AttentionEncoder(**encoder_parameter)
            decoder = AttentionDecoder(**decoder_parameter)
            model = Seq2SeqWithAttention(encoder, decoder, self.max_sequence_size)
        elif self.nmt_model == 'transformer':
            encoder = TransformerEncoder(**encoder_parameter)
            decoder = TransformerDecoder(**decoder_parameter)
            model = Transformer(encoder, decoder)
        else:
            print("적절하지 않은 model 이름입니다. basic_parameter의 model변수를 확인해주세요")
            raise TypeError
        return model

    def get_train_loader(self):
        train_dataset = TrainDataset(self.x_train_path, self.y_train_path, self.ko_voc, self.en_voc,
                                     self.max_sequence_size)
        point_sampler = torch.utils.data.RandomSampler(train_dataset)  # data의 index를 반환하는 함수, suffle를 위한 함수
        # dataset을 인자로 받아 data를 뽑아냄
        train_loader = DataLoader(train_dataset, batch_size=self.basic_parameter.batch_size, sampler=point_sampler)
        return train_loader

    def get_val_loader(self):
        val_dataset = TrainDataset(self.x_val_path, self.y_val_path, self.ko_voc, self.en_voc,
                                   self.max_sequence_size)
        point_sampler = torch.utils.data.RandomSampler(val_dataset)  # data의 index를 반환하는 함수, suffle를 위한 함수
        # dataset을 인자로 받아 data를 뽑아냄
        val_loader = DataLoader(val_dataset, batch_size=self.basic_parameter.batch_size, sampler=point_sampler)
        return val_loader
