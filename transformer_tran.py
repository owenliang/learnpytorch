'''
基于transformer的翻译模型 - pytorch教程
https://pytorch.org/tutorials/beginner/translation_transformer.html
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from timeit import default_timer as timer
''' 
第一部分：词表生成
'''

# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
# de和en各自的分词器
token_transform = {}
# de和en各自的词表
vocab_transform = {}

# 分词器
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# helper function to yield list of tokens
# 句子分词
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        # 对句子分词
        yield token_transform[language](data_sample[language_index[language]])

# 特殊字符（未知,填充,开始,结束)
# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# 从训练数据中, 分别生成en和de的词表
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    # 训练集
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    # 构建词表
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

'''
 第二部分：翻译模型的定义
'''

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
# 位置编码，会在模型里加到词向量上面
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,         # 每个词的位置向量宽度
                 dropout: float,    # dropout率
                 maxlen: int = 5000): # 最多5000个词输入
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size) # 生成emb_size/2宽的向量
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) # 为每个词生成序号，竖起来
        pos_embedding = torch.zeros((maxlen, emb_size)) # 词向量矩阵
        pos_embedding[:, 0::2] = torch.sin(pos * den) # 每个词向量的偶数位置填充sin
        pos_embedding[:, 1::2] = torch.cos(pos * den) # 每个词向量的奇数位置填充cos
        pos_embedding = pos_embedding.unsqueeze(-2) # 加上batch维度?

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :]) # 输入的token向量序列和对应位置的pos向量相加

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
# 词id序列转emb序列
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size) 
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size) # 词id序列分别查emb向量,还elem-wise乘了一下sqrt emb size

# Seq2Seq Network
# 序列生成模型
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        # transofmer is all you need 模型，encoder+decoder架构
        self.transformer = Transformer(d_model=emb_size, # 词向量宽度
                                       nhead=nhead, # 注意力多头个数
                                       num_encoder_layers=num_encoder_layers, # encoder阶段堆叠
                                       num_decoder_layers=num_decoder_layers, # decoder阶段堆叠
                                       dim_feedforward=dim_feedforward, # addnorm层神经元个数
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size) # 下一个词预测, 输入emb pooling, 输出神经元个数是翻译目标语言的词表大小
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size) # 输入词id emb
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size) # 输出词id emb
        self.positional_encoding = PositionalEncoding( # 输入pos
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src)) # 输入词Id序列->输入词emb序列->引入pos信息
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg)) # 输出词id序列->输出词emb序列->引入pos信息
        # encoder->decoder->
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, # src_mask,tgt_mask是屏蔽注意力用的attention_mask
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)  # src_padding_mask和tgt_padding_mask是忽略padding位置用的padding mask
        # decoder输出emb宽的pooling向量, 再过linear转词概率预测
        return self.generator(outs)

    # 推理时encoder阶段
    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    # 推理时decoder阶段
    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


'''
 第三部分：准备训练阶段
'''

# 掩码是加到attention score上面的，这样-inf加上去就导致softmax为0，起到了忽略输入的效果
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1) # 上三角矩阵标1
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) # 标0的转-inf float，其他标0
    return mask

# 参数是词id序列，返回模型所需的mask(attention mask和padding mask)
def create_mask(src, tgt):
    src_seq_len = src.shape[0] # 输入序列的词个数
    tgt_seq_len = tgt.shape[0] # 输出序列的词个数

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len) # 输出序列的mask，要mask住每个词后面的部分
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool) # 输入序列的mask，是全0的，不做任何mask

    # PAD填充位置填1,其他填0
    src_padding_mask = (src == PAD_IDX).transpose(0, 1) 
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE]) # de输入词表大小
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE]) # en输出词表大小
EMB_SIZE = 512 # 词向量宽度
NHEAD = 8   # 自注意力多头个数
FFN_HID_DIM = 512   # feedforward保持输出emb宽
BATCH_SIZE = 128    
NUM_ENCODER_LAYERS = 3 # encoder堆叠三层encoder
NUM_DECODER_LAYERS = 3 # decoder堆叠三层decoder

# 定义seq2seq模型
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
# 所有模型参数
for p in transformer.parameters():
    if p.dim() > 1: 
        nn.init.xavier_uniform_(p)  # 均匀分布初始化参数初始值

# 模型放到GPU上
transformer = transformer.to(DEVICE)

# linear预测的下一个词概率和真实下一个词求损失
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# 优化器
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
# 准备de和en两种句子的预处理方法, 即构造流水线: 分词->id化->添加[BOS]和[EOS]
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
# 输入1批样本
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    # 对于每一对de句子和en句子
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))) # de的句子id序列 
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n"))) # en的句子id序列

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)    # 这一batch的输入句子对齐长度,返回(seq_size,batch_size)的形状
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)    # 这一batch的输出句子对齐长度,返回(seq_size,batch_size)的形状
    return src_batch, tgt_batch

''' 
 第四部分：开始训练
''' 

def train_epoch(model, optimizer):
    model.train() # 训练状态(dropout生效)
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)) # 数据集
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn) # 数据迭代器

    for src, tgt in train_dataloader:
        # 样本放入GPU
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        # src和tgt是(seq_len,batch_size)形状的
        print('src size:', src.size(), 'tgt size:', tgt.size()) # src size: torch.Size([27, 128]) tgt size: torch.Size([24, 128])

        # 最后1个token不参与model forward
        tgt_input = tgt[:-1, :]
        print('tgt_input size', tgt_input.size()) # tgt_input size torch.Size([23, 128])

        # 生成encoder pad mask和decoder注意力mask
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        print('src_mask size:', src_mask.size(), 'tgt_mask size:', tgt_mask.size()) # src_mask size: torch.Size([27, 27]) tgt_mask size: torch.Size([23, 23])
        print('src_padding_mask size:', src_padding_mask.size(), 'tgt_padding_mask size:', tgt_padding_mask.size()) # src_padding_mask size: torch.Size([128, 27]) tgt_padding_mask size: torch.Size([128, 23])
        # print('src_mask:',src_mask)
        # print('tgt_mask:', tgt_mask)

        # forward，依据每个样本[0,tgt_size)位置的token，预测出[1,tgt_size]位置的token
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        print('logits size:', logits.size()) # logits size: torch.Size([23, 128, 10837]), (词序列长度,批大小,词表大小) -> (每个句子23个词,128个句子,每个词有10837种可能)

        optimizer.zero_grad()
        
        # 计算每个样本[1,tgt_size]这些token id和预测出的[1,tgt_size]位置token概率的误差
        tgt_out = tgt[1:, :]
        print('tgt_out size:', tgt_out.size())
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)) # 把batch维直接去掉,这样就是每个样本的每个token的logis和每个样本的每个token id一一对应求loss
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))

NUM_EPOCHS = 16
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    #val_loss = evaluate(transformer)
    val_loss=0
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


''' 
 第五部分：推理
'''
# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
# 翻译方法, 输入de句子
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval() # 推理模式(dropout关闭)
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1) # src是(seq_size,batch_size),和train时候的dim顺序一样
    print('tranlate src:',src.shape)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    print('src_mask src:',src_mask.shape)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))