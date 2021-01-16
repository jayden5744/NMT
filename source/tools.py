# -*- coding:utf-8 -*-
import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from source.Utils.utils import Helper
from source.Utils.bleu import n_gram_precision
from source.data_helper import TrainDataset
from source.Utils.early_stopping import EarlyStopping
from source.Utils.cross_entropy import CrossEntropyLoss
from source.Utils.optimizer import InverseSqrt

from source.Models.seq2seq import Seq2SeqEncoder, Seq2SeqDecoder, Seq2Seq
from source.Models.seq2seq_attention import AttentionEncoder, AttentionDecoder, Seq2SeqWithAttention
from source.Models.transformer import TransformerEncoder, TransformerDecoder, Transformer, greedy_decoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        self.criterion = CrossEntropyLoss(ignore_index=self.trc_voc['<pad>'], smooth_eps=self.label_smoothing,
                                          from_logits=False)
        self.writer = SummaryWriter()  # tensorboard 기록
        self.early_stopping = EarlyStopping(patience=self.early_stopping, verbose=True,
                                            model_path=self.model_path, nmt_model=self.nmt_model)
        self.train()  # train 실행

    def train(self):
        start = time.time()  # 모델 시작 시간 기록
        encoder_parameter = self.encoder_parameter()  # encoder parameter
        decoder_parameter = self.decoder_parameter()  # decoder parameter
        model = self.select_model(encoder_parameter, decoder_parameter)
        model = nn.DataParallel(model).cuda()
        model.to(device)
        model.train()

        print(f'The model has {count_parameters(model):,} trainable parameters')
        model.apply(initialize_weights)

        # optimizer
        optimizer = InverseSqrt(self.basic_parameter['encoder_vocab_size'], warmup_end_lr=self.basic_parameter['learning_rate'],
                                    optimizer=opt.Adam(model.parameters(), lr=self.basic_parameter['learning_rate'],
                                                       betas=(0.9, 0.98), weight_decay=0.0001))

        epoch_step = len(self.train_loader) + 1  # 전체 데이터 셋 / batch_size
        total_step = self.basic_parameter['epochs'] * epoch_step  # 총 step 수

        if self.nmt_model == 'seq2seq':
            train_ratios = cal_teacher_forcing_ratio(self.seq2seq['learning_method'], total_step)  # train learning method

        elif self.nmt_model == 'attention':
            train_ratios = cal_teacher_forcing_ratio(self.attention['learning_method'], total_step) # train learning method

        step = 0

        for epoch in range(self.basic_parameter['epochs']):            # 매 epoch 마다
            for i, data in enumerate(self.train_loader, 0):         # train에서 data를 불러옴
                try:
                    optimizer.optimizer.zero_grad()     # optimizer 모든 변화도 0
                    src_input, trc_input, trc_output = data
                    if self.nmt_model == 'seq2seq':
                        output = model(src_input, trc_input, teacher_forcing_rate=train_ratios[step])
                    elif self.nmt_model == 'attention':
                        output, attention = model(src_input, trc_input, teacher_forcing_rate=train_ratios[step])
                    elif self.nmt_model == 'transformer':
                        output, attention = model(src_input, trc_input)
                    else:
                        raise ValueError("해당 nmt_model은 존재하지 않습니다.")
                    loss, accuracy, ppl = self.cal_loss_acc_per(output, trc_output)

                    # Training Log
                    if step % self.train_step_print == 0:
                        self.writer.add_scalar('train/loss', loss.item(), step)  # save loss to tensorboard
                        self.writer.add_scalar('train/accuracy', accuracy.item(), step)  # save accuracy to tb
                        self.writer.add_scalar('train/PPL', ppl, step)  # save Perplexity to tb

                        print('[Train] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                              '=>  loss : {5:10f}  accuracy : {6:12f}  PPL : {7:6f}'
                              .format(epoch, i, epoch_step, step, total_step, loss.item(), accuracy.item(), ppl))

                    # Validation Log
                    if step % self.val_step_print == 0: # validation step마다
                        with torch.no_grad():  # validation은 학습되지 않음
                            model.eval()  # 모델을 평가상태로
                            val_loss, val_accuracy, val_ppl, val_bleu = self.val_step(model)
                            self.writer.add_scalar('val/loss', val_loss, step)  # save loss to tb
                            self.writer.add_scalar('val/accuracy', val_accuracy, step)  # save accuracy to tb
                            self.writer.add_scalar('val/PPL', val_ppl, step)  # save PPl to tb
                            self.writer.add_scalar('val/BLEU', val_bleu, step)  # save BLEU to tb
                            print('[Val] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                                  '=>  loss : {5:10f}  accuracy : {6:12f}   PPL : {7:10f}'
                                  .format(epoch, i, epoch_step, step, total_step, val_loss, val_accuracy, val_ppl))
                            self.early_stopping(val_loss, model, epoch, step, self.encoder_parameter(),
                                                self.decoder_parameter(), self.max_sequence_size)
                            model.train()   # 모델을 훈련상태로

                    # Save Model Point
                    if step % self.step_save == 0:  # save step마다
                        print("time :", time.time() - start)  # 걸린시간 출력
                        self.save_model(model=model, epoch=epoch, step=step)
                    if self.early_stopping.early_stop:
                        print("Early Stopping")
                        raise KeyboardInterrupt
                    loss.backward()  # 역전파 단계
                    optimizer.step()  # encoder 매개변수 갱신
                    step += 1

                # If KeyBoard Interrupt Save Model
                except KeyboardInterrupt:
                    self.save_model(model=model, epoch=epoch, step=step)


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
        train_dataset = TrainDataset(self.x_train_path, self.y_train_path, self.src_voc, self.trc_voc,
                                     self.max_sequence_size)
        point_sampler = torch.utils.data.RandomSampler(train_dataset)  # data의 index를 반환하는 함수, suffle를 위한 함수
        # dataset을 인자로 받아 data를 뽑아냄
        train_loader = DataLoader(train_dataset, batch_size=self.basic_parameter['batch_size'], sampler=point_sampler)
        return train_loader

    def get_val_loader(self):
        val_dataset = TrainDataset(self.x_val_path, self.y_val_path, self.src_voc, self.trc_voc,
                                   self.max_sequence_size)
        point_sampler = torch.utils.data.RandomSampler(val_dataset)  # data의 index를 반환하는 함수, suffle를 위한 함수
        # dataset을 인자로 받아 data를 뽑아냄
        val_loader = DataLoader(val_dataset, batch_size=self.basic_parameter['batch_size'], sampler=point_sampler)
        return val_loader

    def save_model(self, model, epoch, step):
        model_name = '{0:06d}_{1}.pth'.format(step, self.nmt_model)
        model_path = os.path.join(self.model_path, model_name)
        torch.save({
            'epoch': epoch,
            'steps': step,
            'seq_len': self.max_sequence_size,
            'encoder_parameter': self.encoder_parameter(),
            'decoder_parameter': self.decoder_parameter(),
            'model_state_dict': model.state_dict()
        }, model_path)

    # calculate loss, accuracy, Perplexity
    def cal_loss_acc_per(self, out, trc):
        # out => [batch_size, sequence_len, vocab_size]
        # tar => [batch_size, sequence_len]
        out = out.view(-1, out.size(-1))
        trc = trc.view(-1).to(device)

        # out => [batch_size * sequence_len, vocab_size]
        # tar => [batch_size * sequence_len]
        loss = self.criterion(out, trc)     # calculate loss with CrossEntropy
        ppl = math.exp(loss.item())         # perplexity = exponential(loss)

        indices = out.max(-1)[1]  # 배열의 최대 값이 들어 있는 index 리턴

        invalid_targets = trc.eq(self.trc_voc['<pad>'])  # tar 에 있는 index 중 pad index가 있으면 True, 없으면 False
        equal = indices.eq(trc)  # target이랑 indices 비교
        total = 0
        for i in invalid_targets:
            if i == 0:
                total += 1
        accuracy = torch.div(equal.masked_fill_(invalid_targets, 0).long().sum().to(dtype=torch.float32), total)
        return loss, accuracy, ppl

    # validation step
    def val_step(self, model):
        total_loss = 0
        total_accuracy = 0
        total_ppl = 0
        with torch.no_grad():  # 기록하지 않음
            count = 0
            for data in self.val_loader:
                src_input, trc_input, trc_output = data
                output = model(src_input, trc_input)

                if isinstance(output, tuple):  # attention이 같이 출력되는 경우 output만
                    output = output[0]
                loss, accuracy, ppl = self.cal_loss_acc_per(output, trc_output)
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_ppl += ppl
                count += 1
            if self.nmt_model == 'transformer':
                test_input = src_input[0].unsqueeze(0)
                greedy_dec_input = greedy_decoder(model, test_input, seq_len=self.max_sequence_size)
                output, _ = model(test_input, greedy_dec_input)
                indices = output.view(-1, output.size(-1)).max(-1)[1].tolist()
            else:
                _, indices = output.view(-1, output.size(-1)).max(-1)
                indices = indices[:self.max_sequence_size].tolist()
            a = src_input[0].tolist()
            b = trc_output[0].tolist()
            output_sentence = self.tensor2sen_trc(indices)
            target_sentence = self.tensor2sen_trc(b)
            bleu_score = n_gram_precision(output_sentence[0], target_sentence[0])
            print("-------test-------")
            print("Korean: ", self.tensor2sen_src(a))  # input 출력
            print("Predicted : ", output_sentence)  # output 출력
            print("Target :", target_sentence)  # target 출력
            print('BLEU Score : ', bleu_score)
            avg_loss = total_loss / count  # 평균 loss
            avg_accuracy = total_accuracy / count  # 평균 accuracy
            avg_ppl = total_ppl / count  # 평균 Perplexity
            return avg_loss, avg_accuracy, avg_ppl, bleu_score

    def tensor2sen_src(self, indices: torch.Tensor) -> list:
        result = []
        trans_sen = []
        for idx in indices:
            word = self.src_voc.IdToPiece(idx)
            if word == '<pad>':  # padding 나오면 끝
                break
            trans_sen.append(word)
        trans_sen = ''.join(trans_sen).replace('▁', ' ').strip()  # sentencepiece 에 _ 제거
        result.append(trans_sen)
        return result


    def tensor2sen_trc(self, indices: torch.Tensor) -> list:
        result = []
        trans_sen = []
        for idx in indices:
            word = self.trc_voc.IdToPiece(idx)
            if word == '</s>':  # End token 나오면 stop
                break
            trans_sen.append(word)
        trans_sen = ''.join(trans_sen).replace('▁', ' ').strip()  # sentencepiece 에 _ 제거
        result.append(trans_sen)
        return result
