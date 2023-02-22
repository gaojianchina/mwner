from fastNLP.embeddings import bert_embedding
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import NERModel
from padder import FixLengthPadder
from data_process import get_ngrams, read_emb_file
from fastNLP.io import WeiboNERPipe, OntoNotesNERPipe, MsraNERPipe, PeopleDailyPipe
from fastNLP.core.batch import DataSetIter
from fastNLP.core.utils import seq_len_to_mask
from fastNLP import Vocabulary, SpanFPreRecMetric
from fastNLP import RandomSampler, SequentialSampler
from configparser import ConfigParser
from logger import Logger
from tokenizer import Tokenizer



def get_data_bundle(data_name):
    """根据数据集名字获得data_bundle."""
    assert data_name.lower() in ["weibo", "resume", "onto", "people_daily", "msra", "resume"]
    data_map = {
        "weibo": WeiboNERPipe,
        "onto": MsraNERPipe, # OntoNotesNERPipe
        "msra": MsraNERPipe,
        "people_daily": PeopleDailyPipe,
        "resume": MsraNERPipe
    }

    if data_name  == "resume":
        path = {
            "train": "dataset/resume/train.char.bio",
            "dev": "dataset/resume/dev.char.bio"
        }
    elif data_name == "onto":
        path = {
            "train": "dataset/onto/onto_train.txt",
            "dev": "dataset/onto/onto_dev.txt"
        }
    else:
        path = None  # 其他三个数据集会自动下载.

    return data_map[data_name]().process_from_file(path)

def get_config():
    """
    获取.cfg文件中的配置
    """
    parser = ConfigParser()
    parser.read("./train.cfg", encoding="utf-8")
    return parser

def get_Ngrams(dataset, cws):
    """生成n-grams."""
    if dataset.has_field('chars'):
        ## 删除chars列
        dataset.delete_field('chars')
    dataset.apply(lambda ins: get_ngrams(ins, 2), new_field_name='2grams')
    dataset.apply(lambda ins: get_ngrams(ins, 3), new_field_name='3grams')
    dataset.apply(lambda ins: get_ngrams(ins, 4), new_field_name='4grams')
    dataset.apply(lambda ins: get_ngrams(ins, 5), new_field_name='5grams')
    tok = Tokenizer(cws)
    dataset.apply(lambda ins: tok.getWordIndices(sentence="".join(ins["raw_chars"])), new_field_name="word_indices")
    dataset.apply(lambda ins: tok.getNgramsIndices(sentence="".join(ins["raw_chars"])), new_field_name="ngram_indices")

    return dataset

def tokenize(vocab, dataset, cws):
    """将dataset中的字符/词映射成index."""
    dataset = get_Ngrams(dataset, cws)

    vocab.index_dataset(dataset, field_name="raw_chars")
    vocab.index_dataset(dataset, field_name="2grams")
    vocab.index_dataset(dataset, field_name="3grams")
    vocab.index_dataset(dataset, field_name="4grams")
    vocab.index_dataset(dataset, field_name="5grams")

    return dataset

def fix_length(dataset, padder):
    """使用padder生成固定长度的tensors."""
    # 利用dataset的set_padder函数设定words field的padder
    dataset.set_padder('raw_chars', padder)
    dataset.set_padder('2grams', padder)
    dataset.set_padder('3grams', padder)
    dataset.set_padder('4grams', padder)
    dataset.set_padder('5grams', padder)

    dataset.set_padder('target', padder)

    return dataset

def set_inputs_and_targets(iter):
    """指定DataSetIter中输入X和标签y."""
    iter.set_input('raw_chars', '2grams', '3grams', '4grams', '5grams', 'word_indices', 'ngram_indices', 'seq_len')  # 设置多个列为input
    iter.set_target('target')
    
    return iter

def to(X, tags, device="cuda", num_of_grams=4):
    """将数据移到device上."""
    # todo: 需要进行批处理
    ngrams = torch.cat((X['2grams'], X['3grams']))
    if num_of_grams == 3:
        ngrams = torch.cat((ngrams, X['4grams']))
    elif num_of_grams == 4:
        ngrams = torch.cat((ngrams, X['4grams'], X['5grams']))
    
    chars = X['raw_chars'].to(device)
    ngrams = ngrams.to(device)
    target = tags["target"].to(device)
    word_indices = X["word_indices"].to(device)
    ngram_indices = X["ngram_indices"].to(device)

    batch_size = chars.shape[0]
    chars = chars.view(batch_size, chars.shape[0]//batch_size, chars.shape[1])
    ngrams = ngrams.view(batch_size, ngrams.shape[0]//batch_size, ngrams.shape[1])
    # word_indices = word_indices.view(batch_size, ngrams.shape[0]//batch_size, ngrams.shape[1])
    # ngram_indices = ngram_indices.view(batch_size, ngrams.shape[0]//batch_size, ngrams.shape[1])

    return (chars, ngrams, word_indices, ngram_indices), target

def train(model, train_iter, optimizer, device):
    """训练模型"""
    model.train()

    losses = 0
    for X, target in train_iter:
        seq_len = X["seq_len"]
        X, target = to(X, target, device)
        
        # 使用fastNLP中的seq_len_to_mask函数生成mask.
        mask = seq_len_to_mask(torch.LongTensor(seq_len), 64).to(device)

        optimizer.zero_grad()

        y = model(X)
        loss = model.get_loss(y, target, mask)
        losses += loss.item()

        loss.backward()
        optimizer.step()
    return losses / len(train_iter)

def evaluate(model, eval_iter, tag_vocab, device):
    """评估模型"""
    model.eval()
    
    loss = 0
    metric = SpanFPreRecMetric(tag_vocab)

    with torch.no_grad():
        for X, target in eval_iter:
            seq_len = X["seq_len"]
            X, target = to(X, target, device)
            # mask = masking(seq_len.item(), 64).to(device)
            mask = seq_len_to_mask(torch.LongTensor(seq_len), 64).to(device)

            y = model(X)
            loss += model.get_loss(y, target, mask).item()
            y = model.predict(y, mask)
            metric.evaluate(y, target, seq_len)
        
        eval_result = metric.get_metric()
    return loss/len(eval_iter), eval_result


def main():
    # 获取config
    config = get_config()

    # 从config中得到参数.
    batch_size = config.getint("common", "batch_size")
    data_name = config.get("common", "dataset")
    tokenizer = config.get("common", "tokenizer")
    log_file_name = data_name + "_" + tokenizer

    logging = Logger.getLogger(file=log_file_name)
    logging.info("dataset: {}, batch_size: {}, tokenizer: {}".format(data_name, batch_size, tokenizer))

    device= config.get("common", "device")
    if device == 'cuda':
        assert torch.cuda.is_available()

    # 加载数据集
    data_bundle = get_data_bundle(data_name)

    # 加载词向量
    word_emb_file = config.get("common", "word_emb_file")
    _, (i2w, _), _ = read_emb_file(word_emb_file)

    # 生成词表和标签
    word_vocab = Vocabulary()
    word_vocab.add_word_lst(i2w)
    target_vocab = data_bundle.get_vocab('target')

    train_dataset = data_bundle.get_dataset('train')
    eval_dataset= data_bundle.get_dataset('dev')
    
    # 对数据集进行增强: 获得ngrams
    train_dataset = tokenize(word_vocab, train_dataset, tokenizer)
    eval_dataset = tokenize(word_vocab, eval_dataset, tokenizer)
    

    # 设置并查看输入和标签
    train_dataset = set_inputs_and_targets(train_dataset)
    eval_dataset = set_inputs_and_targets(eval_dataset)
    # print(train_dataset.print_field_meta())
    # print(eval_dataset.print_field_meta())

    # 设定FixLengthPadder的固定长度为max_word_len
    max_len = config.getint("common", "max_word_len")
    fix_len_padder = FixLengthPadder(pad_val=word_vocab.padding_idx, length=max_len)
    train_dataset = fix_length(train_dataset, fix_len_padder)
    eval_dataset = fix_length(eval_dataset, fix_len_padder)

    # 迭代器DataSetIter
    train_sampler = RandomSampler()
    eval_sampler = SequentialSampler()

    train_iter = DataSetIter(train_dataset, batch_size=batch_size, sampler=train_sampler)
    eval_iter = DataSetIter(eval_dataset, batch_size=batch_size, sampler=eval_sampler)

    # 模型
    num_of_grams = config.getint("common", "num_of_grams")
    model = NERModel(word_vocab, target_vocab=target_vocab, num_of_grams=num_of_grams, word_emb_file=word_emb_file, is_first=False)
    model.to(device)
    
    # 学习率和优化器
    lr = config.getfloat("common", "lr")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    best_f1 = 0
    is_save = config.getboolean("common", "is_save")  # 是否保存模型
    num_epochs = config.getint("common", "num_epochs")

    logging.info("{} epochs totally. start training...".format(num_epochs))
    for epoch in range(num_epochs):
        train_loss = train(model, train_iter, optimizer, device)
        
        if epoch % 1 == 0:
            eval_loss, result = evaluate(model, eval_iter, target_vocab, device)
            logging.info("the {}-th epoch, train_loss: {}, eval_loss: {}, metrics: {}".format(
                epoch, train_loss, eval_loss, result))
            
            if is_save and result["f1"] > best_f1:
                best_f1 = result["f1"]
                torch.save(model.state_dict(), 'models/best_params.pkl')
                # model.load_state_dict(torch.load('models/best_params.pkl'))


if __name__ == "__main__":
    main()