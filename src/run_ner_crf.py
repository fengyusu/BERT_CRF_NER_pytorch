# -*- coding: utf-8 -*-

import gc
import time
import warnings
from argparse import ArgumentParser

import numpy as np

from src.utils.bert_utils import *
from src.utils.crf_utils.ner_utils import *
from src.utils.crf_utils.tag_label import TagsLabel
from src.utils.utils import save_pickle, load_pkl, load_file, save_pkl

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

def read_data(args, tokenizer, label_vocab):

    train_inputs, dev_inputs = defaultdict(list), defaultdict(list)

    with open(args.train_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            words, labels = line.strip('\n').split('\t')
            text = words.split('\002')
            label = labels.split('\002')
            build_bert_inputs(train_inputs, label, text, tokenizer, label_vocab)

    with open(args.dev_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            words, labels = line.strip('\n').split('\t')
            text = words.split('\002')
            label = labels.split('\002')
            build_bert_inputs(dev_inputs, label, text, tokenizer, label_vocab)

    train_cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')
    dev_cache_pkl_path = os.path.join(args.data_cache_path, 'dev.pkl')

    save_pickle(train_inputs, train_cache_pkl_path)
    save_pickle(dev_inputs, dev_cache_pkl_path)

def display(train_loss, evaluate_scores):

    train_loss_trace = go.Scatter(
        x = np.arange(len(train_loss)),
        y = np.array(train_loss)[:,0],
        mode='lines + markers + text',
        name='train_loss',
        text=np.array(train_loss)[:,0],
        textposition="top center"
    )
    train_loss_trace1 = go.Scatter(
        x=np.arange(len(train_loss)),
        y=np.array(train_loss)[:, 1],
        mode='lines + markers + text',
        name='tokenize_loss',
        text=np.array(train_loss)[:,1],
        textposition="top center"
    )
    train_loss_trace2 = go.Scatter(
        x=np.arange(len(train_loss)),
        y=np.array(train_loss)[:, 2],
        mode='lines + markers + text',
        name='entity_loss',
        text=np.array(train_loss)[:,2],
        textposition="top center"
    )
    evaluate_loss_trace = go.Scatter(
        x = np.arange(len(evaluate_scores)),
        y = np.array(evaluate_scores)[:,-1],
        mode = 'lines + markers + text',
        name = 'evaluate_loss',
        text=np.array(evaluate_scores)[:,-1],
        textposition="top center"
    )

    loss_data = [train_loss_trace, train_loss_trace1, train_loss_trace2, evaluate_loss_trace]
    loss_layout = dict(title='loss ', xaxis=dict(title='step'), yaxis=dict(title='value'))
    loss_fig = dict(data=loss_data, layout=loss_layout)
    py.iplot(loss_fig, filename='loss-line')

    evaluate_f1_trace = go.Scatter(
        x=np.arange(len(evaluate_scores)),
        y=np.array(evaluate_scores)[:, 0],
        mode='lines + markers + text',
        name='f1 score',
        text=np.array(evaluate_scores)[:, 0],
        textposition="top center"
    )
    evaluate_precision_trace = go.Scatter(
        x=np.arange(len(evaluate_scores)),
        y=np.array(evaluate_scores)[:, 1],
        mode='lines + markers + text',
        name='precision',
        text=np.array(evaluate_scores)[:, 1],
        textposition="top center"
    )
    evaluate_recall_trace = go.Scatter(
        x=np.arange(len(evaluate_scores)),
        y=np.array(evaluate_scores)[:, 2],
        mode='lines + markers + text',
        name='recall',
        text=np.array(evaluate_scores)[:,2],
        textposition="top center"
    )
    score_data = [evaluate_f1_trace, evaluate_precision_trace, evaluate_recall_trace]
    score_layout = dict(title='score ', xaxis=dict(title='step'), yaxis=dict(title='value'))
    score_fig = dict(data=score_data, layout=score_layout)
    py.iplot(score_fig, filename='score-line')

    time.sleep(1)

def eval(args, model, dev_dataloader, label_vocab, tokenizer, best_f1, ema):

    model.eval()

    if args.use_ema and ema is not None:
        if args.ema_start:
            ema.apply_shadow()

    print("\n >> Start evaluating ... ... ")

    metric, entity_info = evaluate(args, model, dev_dataloader, label_vocab)

    f1_score, precision, recall = metric['f1'], metric['precision'], metric['recall']
    dev_loss = metric['avg_dev_loss']

    f1_score, precision, recall, dev_loss = round(f1_score, 4), round(precision, 4), \
                                            round(recall, 4), round(dev_loss, 4)

    print(f"\n >> cur step model ."
          f"\n >> f1 : {f1_score}, precision : {precision}, recall : {recall}, "
          f"dev loss : {dev_loss} .")

    if f1_score > best_f1:
        best_f1 = f1_score
        save_model(args, model, tokenizer)

        print(f" best model saved.")

        if args.print_entity_info:
            print("***** Entity results %s *****")
            for key in sorted(entity_info.keys()):
                print("******* %s results ********" % key)
                info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
                print(info)

    if args.use_ema and ema is not None:
        if args.ema_start:
            ema.restore()

    model.train()

    return best_f1, metric

def train(args):

    label_vocab = TagsLabel()

    num_labels = label_vocab.get_item_size()

    tokenizer, model = build_model_and_tokenizer(args, num_labels)
    model.to(args.device)

    if not os.path.exists(os.path.join(args.data_cache_path, 'train.pkl')):
        read_data(args, tokenizer, label_vocab)

    train_dataloader, dev_dataloader = load_data(args, tokenizer)

    total_steps = args.num_epochs * len(train_dataloader)
    optimizer, scheduler = build_optimizer(args, model, total_steps)

    global_steps, total_loss, cur_avg_loss, cur_avg_tokenize_loss, cur_avg_entity_loss, best_f1 = 0, 0., 0., 0., 0., 0.

    train_loss = []
    evaluate_loss = []
    evaluate_scores = []
    print("\n >> Start training ... ... ")
    for epoch in range(1, args.num_epochs + 1):

        train_iterator = tqdm(train_dataloader, desc=f'Epoch_{epoch}', total=len(train_dataloader))

        model.train()

        for batch in train_iterator:

            # best_f1,_ = eval(args, model, dev_dataloader, label_vocab, tokenizer, best_f1=best_f1, ema=None)

            model.zero_grad()

            batch_cuda = batch2cuda(args, batch)
            tokenize_loss, entity_loss = model(**batch_cuda)[0:2]
            if epoch > 0:
                loss = tokenize_loss + entity_loss
            else:
                loss = tokenize_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if args.use_fgm:
                model.zero_grad()
                fgm = FGM(args, model)
                fgm.attack()
                adv_loss = model(**batch_cuda)[0]
                adv_loss.backward()
                fgm.restore()

            if args.use_pgd:
                model.zero_grad()
                pgd = PGD(args, model)
                pgd.backup_grad()
                for t in range(args.adv_k):
                    pgd.attack(is_first_attack=(t == 0))
                    if t != args.adv_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    adv_loss = model(**batch_cuda)[0]
                    adv_loss.backward()
                pgd.restore()

            total_loss += loss.item()
            cur_avg_loss += loss.item()
            cur_avg_tokenize_loss += tokenize_loss.item()
            cur_avg_entity_loss += entity_loss.item()

            optimizer.step()
            scheduler.step()

            if args.use_ema:
                if args.ema_start:
                    ema.update()

            optimizer.zero_grad()

            if (global_steps + 1) % args.logging_steps == 0:

                cur_avg_loss = cur_avg_loss / args.logging_steps
                cur_avg_tokenize_loss = cur_avg_tokenize_loss / args.logging_steps
                cur_avg_entity_loss = cur_avg_entity_loss / args.logging_steps
                global_avg_loss = total_loss / (global_steps + 1)

                print(f"\n>> epoch - {epoch},  global steps - {global_steps + 1}, "
                      f"epoch avg loss - {cur_avg_loss:.4f}, global avg loss - {global_avg_loss:.4f}.")

                if args.use_ema:
                    if global_steps >= args.ema_start_step and not args.ema_start:
                        print('\n>>> EMA starting ...')
                        args.ema_start = True
                        ema = EMA(model.module if hasattr(model, 'module') else model, decay=0.999)


            global_steps += 1
            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')

        train_loss.append([cur_avg_loss, cur_avg_tokenize_loss, cur_avg_entity_loss])
        cur_avg_loss = 0.
        cur_avg_tokenize_loss = 0.
        cur_avg_entity_loss = 0.

        if args.do_eval:
            best_f1, metric = eval(args, model, dev_dataloader, label_vocab, tokenizer, best_f1, ema=None)
            evaluate_scores.append([metric['f1'], metric['precision'], metric['recall'], metric['avg_dev_loss']])

    if args.use_ema:
        ema.apply_shadow()

    if not args.do_eval:
        save_model(args, model, tokenizer)

    if args.display:
        display(train_loss, evaluate_scores)

    data = time.asctime(time.localtime(time.time())).split(' ')
    now_time = data[-1] + '-' + data[-5] + '-' + data[-3] + '-' + \
    data[-2].split(':')[0] + '-' + data[-2].split(':')[1] + '-' + data[-2].split(':')[2]
    os.makedirs(os.path.join(args.output_path, f'f1-{best_f1}-{now_time}'), exist_ok=True)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

    print('\n >> Finish training .')


def main(ner_type):
    parser = ArgumentParser()

    parser.add_argument('--ner_type', type=str, default=ner_type)

    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--output_path', type=str,
                        default=f'../user_data/output_model/{ner_type}')
    parser.add_argument('--train_path', type=str,
                        default=f'../raw_data/train.json')
    parser.add_argument('--dev_path', type=str,
                        default=f'../raw_data/dev.json')
    parser.add_argument('--data_cache_path', type=str,
                        default=f'../user_data/process_data/pkl/{ner_type}')
    parser.add_argument("--label_file", type=str,
                        default="../raw_data/label.txt")
    parser.add_argument('--model_name_or_path', type=str,
                        default=f'../user_data/pretrain_model/bert-base-chinese')

    parser.add_argument('--do_lower_case', type=bool, default=True)

    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--print_entity_info', type=bool, default=False)
    parser.add_argument('--display', type=bool, default=True)

    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=100)

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--crf_learning_rate', type=float, default=3e-2)
    parser.add_argument('--classifier_learning_rate', type=float, default=3e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--use_fgm', type=bool, default=False)
    parser.add_argument('--use_pgd', type=bool, default=False)
    parser.add_argument('--use_ema', type=bool, default=False)
    parser.add_argument('--use_lookahead', type=bool, default=False)

    parser.add_argument('--adv_k', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--emb_name', type=str, default='word_embeddings.')

    parser.add_argument('--ema_start', type=bool, default=False)
    parser.add_argument('--ema_start_step', type=int, default=0)

    parser.add_argument('--logging_steps', type=int, default=50)

    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--device', type=str, default='cuda')

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    path_list = [args.output_path, args.data_cache_path]
    for i in path_list:
        os.makedirs(i, exist_ok=True)

    seed_everything(args.seed)

    train(args)


if __name__ == '__main__':
    main('crf')
