import os
import torch
import shutil
import sentencepiece as spm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def create_or_get_voca(save_path, src_corpus_path=None, trc_corpus_path=None, src_vocab_size=4000, trc_vocab_size=4000,
                       src_region="ko", trc_region="en"):
    src_corpus_prefix = '{0}_corpus_{1}'.format(src_region, src_vocab_size)  # vocab_size를 바꾸면 embedding_size도 변경
    trc_corpus_prefix = '{0}_corpus_{1}'.format(trc_region, trc_vocab_size)

    if src_corpus_path and trc_corpus_path:
        templates = '--input={} --model_prefix={} --vocab_size={} ' \
                    '--bos_id=0 --eos_id=1 --unk_id=2 --pad_id=3'
        # input : 학습시킬 텍스트의 위치
        # model_prefix : 만들어질 모델 이름
        # vocab_size : 사전의 크기
        src_model_train_cmd = templates.format(src_corpus_path, src_corpus_prefix, src_vocab_size)
        trc_model_train_cmd = templates.format(trc_corpus_path, trc_corpus_prefix, trc_vocab_size)

        spm.SentencePieceTrainer.Train(src_model_train_cmd)  # Korean 텍스트를 가지고 학습
        spm.SentencePieceTrainer.Train(trc_model_train_cmd)  # English 텍스트를 가지고 학습
        print(os.getcwd())
        print(save_path)
        # 파일을 저장위치로 이동
        shutil.move(src_corpus_prefix + '.model', save_path)
        shutil.move(src_corpus_prefix + '.vocab', save_path)
        shutil.move(trc_corpus_prefix + '.model', save_path)
        shutil.move(trc_corpus_prefix + '.vocab', save_path)

    src_sp = spm.SentencePieceProcessor()
    trc_sp = spm.SentencePieceProcessor()
    src_sp.load(os.path.join(save_path, src_corpus_prefix + '.model'))  # model load
    trc_sp.load(os.path.join(save_path, trc_corpus_prefix + '.model'))  # model load
    return src_sp, trc_sp


class TrainDataset(Dataset):
    def __init__(self, x_path, y_path, ko_voc, en_voc, sequence_size):
        self.x = open(x_path, 'r', encoding='utf-8').readlines()  # korean data 위치
        self.y = open(y_path, 'r', encoding='utf-8').readlines()  # English data 위치
        self.ko_voc = ko_voc  # Korean 사전
        self.en_voc = en_voc  # English 사전
        self.sequence_size = sequence_size  # sequence 최대길이
        self.KO_PAD_ID = ko_voc['<pad>']  # 3 Padding
        self.EN_PAD_ID = en_voc['<pad>']  # 3 Padding
        self.EN_BOS_ID = en_voc['<s>']  # 0 Start Token
        self.EN_EOS_ID = en_voc['</s>']  # 1 End Token

    def __len__(self):  # data size를 넘겨주는 파트
        if len(self.x) != len(self.y):
            raise IndexError('not equal x_path, y_path line size')
        return len(self.x)

    def __getitem__(self, idx):
        encoder_input = self.encoder_input_to_vector(self.x[idx])
        decoder_input = self.decoder_input_to_vector(self.y[idx])
        decoder_output = self.decoder_output_to_vector(self.y[idx])
        return encoder_input, decoder_input, decoder_output

    def encoder_input_to_vector(self, sentence: str):
        idx_list = self.ko_voc.EncodeAsIds(sentence)  # str -> idx
        idx_list = self.padding(idx_list, self.KO_PAD_ID)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def decoder_input_to_vector(self, sentence: str):
        idx_list = self.en_voc.EncodeAsIds(sentence)  # str -> idx
        idx_list.insert(0, self.EN_BOS_ID)  # Start Token 삽입
        idx_list = self.padding(idx_list, self.EN_PAD_ID)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def decoder_output_to_vector(self, sentence: str):
        idx_list = self.en_voc.EncodeAsIds(sentence)  # str -> idx
        idx_list.append(self.EN_EOS_ID)  # End Token 삽입
        idx_list = self.padding(idx_list, self.EN_PAD_ID)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def padding(self, idx_list, padding_id):
        length = len(idx_list)  # 리스트의 길이
        if length < self.sequence_size:
            # sentence가 sequence_size가 작으면 나머지를 padding token으로 채움
            idx_list = idx_list + [padding_id for _ in range(self.sequence_size - len(idx_list))]
        else:
            idx_list = idx_list[:self.sequence_size]
        return idx_list


class TestDataset(Dataset):
    def __init__(self, x_path, ko_voc, sequence_size):
        self.x = open(x_path, 'r', encoding='utf-8').readlines()  # korean data 위치
        self.ko_voc = ko_voc                # Korean 사전
        self.sequence_size = sequence_size  # sequence 최대길이
        self.KO_PAD_ID = ko_voc['<pad>']    # 3 Padding
        self.EN_EOS_ID = ko_voc['</s>']     # 1 End Token
        self.x = open(x_path, 'r', encoding='utf-8').readlines()  # korean data 위치
        self.ko_voc = ko_voc                # Korean 사전
        self.sequence_size = sequence_size  # sequence 최대길이

    def __len__(self):  # data size를 넘겨주는 파트
        return len(self.x)

    def __getitem__(self, idx):
        encoder_input = self.encoder_input_to_vector(self.x[idx])
        return encoder_input

    def encoder_input_to_vector(self, sentence: str):
        idx_list = self.ko_voc.EncodeAsIds(sentence)  # str -> idx
        idx_list.append(self.EN_EOS_ID)  # End Token 삽입
        idx_list = self.padding(idx_list, self.KO_PAD_ID)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def padding(self, idx_list, padding_id):
        length = len(idx_list)  # 리스트의 길이
        if length < self.sequence_size:
            # sentence가 sequence_size가 작으면 나머지를 padding token으로 채움
            idx_list = idx_list + [padding_id for _ in range(self.sequence_size - len(idx_list))]
        else:
            idx_list = idx_list[:self.sequence_size]
        return idx_list


def split_data(src_path, trc_path, val_size=10000, test_size=10000):
    with open(src_path, 'r', encoding='utf-8-sig') as f:
        src_lst = []
        for i in f.readlines():
            src_lst.append(i)

    with open(trc_path, "r", encoding='utf-8-sig') as f:
        trc_lst = []
        for i in f.readlines():
            trc_lst.append(i)

    x_train, x_test, y_train, y_test = train_test_split(src_lst, trc_lst, test_size=test_size, random_state=1111)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=1111)

    return x_train, x_val, x_test, y_train, y_val, y_test

def save_txt(file_name, lst):
    with open(file_name, 'w', encoding='utf-8-sig') as f:
        for i in lst:
            f.write(i)


if __name__ =='__main__':
    src_path = "../Data/ko_total.txt"
    trc_path = "../Data/en_total.txt"
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(src_path, trc_path, val_size=10000, test_size=50000)
    save_txt("../Data/ko.train", x_train)
    save_txt("../Data/ko.val", x_val)
    save_txt("../Data/ko.test", x_test)
    save_txt("../Data/en.train", y_train)
    save_txt("../Data/en.val", y_val)
    save_txt("../Data/en.test", y_test)
