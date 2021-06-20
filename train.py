# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

from sklearn.metrics import f1_score
import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import Bert_CRF, BertForTokenClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup

from data.dataloader import ArgumentProcessor, convert_examples_to_features

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def parsing():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default='arg_parsing', type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    #parser.add_argument("--temp_dir", default=None, type=str, required=True,
    #                    help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--pred_name", default='./pred', type=str, required=True,
                        help="file name to save the output.")
    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",	default=480, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=400, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=9e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=4.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_steps", default=20, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=32,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=16,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")

    args = parser.parse_args()
    return args

def load_and_cache_examples(args, task, tokenizer, mode='dev'):
    if(args.local_rank not in [-1, 0] and mode=='train'):
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = ArgumentProcessor()
    output_mode = args.output_mode
    # Load data features from cache or dataset file
    if(os.path.isfile(args.data_dir)):
        data_dir = './cache/'
        last = '_'.join(args.data_dir.split('/')[-3:])
        #last = args.data_dir.split('/')[-1]
        cached_features_file = data_dir + '/cached_{}_{}_{}_{}'.format(
            mode,
            str(args.max_seq_length),
            str(task), last)
    else:
        cached_features_file = args.data_dir + '_cached_{}_{}_{}'.format(
            mode,
            list(filter(None, args.bert_model.split('/'))).pop(),
            str(args.max_seq_length),
            )

    logger.info("save to %s", cached_features_file)
    if(os.path.exists(cached_features_file)):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        if(mode == 'train'):
            examples = processor.get_train_examples(args.data_dir)
        elif(mode == 'dev'):
            examples = processor.get_dev_examples(args.data_dir)
        elif(mode == 'test'):
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(examples, tokenizer,
                                        max_seq_length=args.max_seq_length,
                                        label_list=processor.get_labels(), output_mode=output_mode,
                                        pad_on_left=False,
                                        pad_token=0,
                                        pad_token_segment_id=0,
                                        mask_padding_with_zero=True,
                                        train_eval = (mode=='train' or mode=='dev'))
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if(args.local_rank == 0 and mode=='train'):
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    # ['input_ids', 'attention_mask', 'crf_mask', 'segment_ids', 'bio_labels', 'type_labels', 'recover']
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    all_crf_mask = torch.tensor([f['crf_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
    all_recover = torch.tensor([f['recover'] for f in features], dtype=torch.long)
    all_index = torch.tensor([f['index'] for f in features], dtype=torch.long)
    if(mode=='train' or mode=='dev'):
        all_bio_labels = torch.tensor([f['bio_labels'] for f in features], dtype=torch.long)
        all_type_labels = torch.tensor([f['type_labels'] for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_crf_mask, all_segment_ids, all_bio_labels, all_type_labels, all_recover, all_index)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_crf_mask, all_segment_ids, all_recover, all_index)

    return dataset

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if(args.max_steps > 0):
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
                )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # ['input_ids', 'attention_mask', 'crf_mask', 'segment_ids', 'bio_labels', 'type_labels', 'recover'
            loss = model(batch[0], token_type_ids=batch[3], attention_mask=batch[1], crf_mask=batch[2],
                                                    bio_labels=batch[4], type_labels=batch[5])

            if(args.n_gpu > 1):
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if(args.gradient_accumulation_steps > 1):
                loss = loss / args.gradient_accumulation_steps

            if(args.fp16):
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if((step + 1) % args.gradient_accumulation_steps == 0):
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if(args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    # Log metrics
                    if(args.local_rank == -1 and args.evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if(args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if(not os.path.exists(output_dir)):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))

                    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(output_dir, CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(output_dir)
                    #logger.info("Saving model checkpoint to %s", output_dir)

            if(args.max_steps > 0 and global_step > args.max_steps):
                epoch_iterator.close()
                break
        if(args.max_steps > 0 and global_step > args.max_steps):
            train_iterator.close()
            break

    if(args.local_rank in [-1, 0]):
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, mode='dev')

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        label_list = args.label_list
        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds, true_labels = {'bio':[],'type':[]}, {'bio':[],'type':[]}
        result = {}
        for input_ids, input_mask, crf_mask, segment_ids, bio_labels, type_labels, re, index  in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            crf_mask = crf_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)
            bio_labels = bio_labels.to(args.device)
            type_labels = type_labels.to(args.device)

            with torch.no_grad():
                # input_ids, word_attention_mask, token_type_ids, sent_attention_mask, label_relation, label_target, label_type
                tmp_eval_loss, tmp_result, logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, crf_mask=crf_mask, bio_labels=bio_labels, type_labels=type_labels, check=True)
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            """
            for key in tmp_result:
                try:
                    result[key] += tmp_result[key]
                except:
                    result[key] = tmp_result[key]
            """
            nb_eval_steps += 1

            bio_logits = logits['bio']
            type_logits = logits['type'].detach().cpu().numpy()

            recovers = re.detach().cpu().numpy()
            bio_labels = bio_labels.detach().cpu().numpy()
            type_labels = type_labels.detach().cpu().numpy()
            seq_lens = crf_mask.sum(-1).detach().cpu().numpy()

            for a in range(5):
                pass
                #print(len(bio_logits[a]), bio_logits[a])
                #print(seq_lens[a], bio_labels[a][:seq_lens[a]])
                #print()

            for bio_logit, type_logit, bio_label, type_label, recover, seq_len in zip(bio_logits, type_logits, bio_labels, type_labels, recovers, seq_lens):

                for key in ['bio', 'type']:
                    preds[key].append([])
                    true_labels[key].append([])

                flag = False
                for b_lo, t_lo, b_l, t_l, r in zip(bio_logit[:seq_len], type_logit, bio_label, type_label, recover):
                    if(r==0):
                        preds['bio'][-1].append(b_lo)
                        true_labels['bio'][-1].append(b_l)

                        if(true_labels['bio'][-1][-1] == 0):
                            true_labels['type'][-1].append(t_l)
                            if(flag==True):
                                preds['type'][-1].append(acc.argmax())

                            acc = t_lo.copy()
                            flag = True
                        elif(true_labels['bio'][-1][-1] == 1):
                            # accumulate value for tpye prediction
                            acc += t_lo.copy()
                            flag = True
                        elif(true_labels['bio'][-1][-1] == 2 and flag == True):
                            preds['type'][-1].append(acc.argmax())
                            flag = False
                else:
                    if(flag==True):
                        preds['type'][-1].append(acc.argmax())


        eval_loss = eval_loss / nb_eval_steps
        #result['expand_f1'] = result['expand_f1'] / nb_eval_steps
        #result['expand_acc'] = result['expand_acc'] / nb_eval_steps

        for key in ['type', 'bio']:
            pred, truth = [], []
            for p, t in zip(preds[key], true_labels[key]):
                pred.extend(p)
                truth.extend(t)
            pred = np.array(pred)
            truth = np.array(truth)
            index = truth>=0
            pred = pred[index]
            truth = truth[index]
            result['{}_f1'.format(key)] = f1_score(y_true=truth, y_pred=pred , average='macro')
            result['{}_acc'.format(key)] = (truth==pred).mean()


        result['eval_loss'] = eval_loss
        result['loss'] = eval_loss
        ###################################################

        torch.save(preds, './pred')
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def test(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    test_task_names = (args.task_name,)
    test_output_dir = args.output_dir

    results = {}
    for test_task in test_task_names:
        test_dataset = load_and_cache_examples(args, test_task, tokenizer, mode='test')

        if(not os.path.exists(test_output_dir) and args.local_rank in [-1, 0]):
            os.makedirs(test_output_dir)

        args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size)

        label_list = args.label_list
        # Eval!
        logger.info("***** Running testing {} *****".format(prefix))
        logger.info("  Num examples = %d", len(test_dataset))
        logger.info("  Batch size = %d", args.test_batch_size)

        preds = {'index':[], 'bio':[],'type':[], 'recover':[], 'loss':[], 'id':[]}
        result = {}
        for input_ids, input_mask, crf_mask, segment_ids, re, indexs  in tqdm(test_dataloader, desc="Evaluating"):
            model.eval()
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            crf_mask = crf_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)

            with torch.no_grad():
                # input_ids, word_attention_mask, token_type_ids, sent_attention_mask, label_relation, label_target, label_type
                logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, crf_mask=crf_mask)

            bio_logits = logits['bio']['output']
            bio_logits_loss = logits['bio']['loss']
            type_logits = logits['type'].detach().cpu().numpy()
            input_ids = input_ids.detach().cpu().numpy()

            recovers = re.detach().cpu().numpy()
            seq_lens = input_mask.sum(-1).detach().cpu().numpy()

            for index, input_id, bio_logit, loss, type_logit, recover, seq_len in zip(indexs, input_ids, bio_logits, bio_logits_loss, type_logits, recovers, seq_lens):
                preds['index'].append(index)

                preds['bio'].append(bio_logit[:seq_len])
                preds['type'].append(type_logit[:seq_len])
                preds['recover'].append(recover[:seq_len])
                preds['id'].append(input_id[:seq_len])

                preds['loss'].append(loss)
        torch.save(preds, args.pred_name)




def main():
    args = parsing()

    if(os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir):
        # raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
        pass
    elif(not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]):
        os.makedirs(args.output_dir)

    if(args.local_rank == -1 or args.no_cuda):
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    if(not args.do_train and not args.do_eval and not args.do_test):
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()
    processor = ArgumentProcessor()
    args.output_mode = "classification"

    label_list = processor.get_labels()
    args.label_list = label_list
    bio_num_labels = len(label_list[0])
    type_num_labels = len(label_list[1])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = Bert_CRF.from_pretrained(args.bert_model, bio_num_labels=bio_num_labels, type_num_labels=type_num_labels)
    #model = BertForTokenClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Training
    if(args.do_train):
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, mode='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    ### Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if(args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0)):
        # Save a trained model, configuration and tokenizer
        logger.info("Saving model checkpoint to %s", args.output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = Bert_CRF.from_pretrained(args.output_dir, bio_num_labels=bio_num_labels, type_num_labels=type_num_labels)
        # model = BertForTokenClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

        # Good practice: save your training arguments together with the trained model
        output_args_file = os.path.join(args.output_dir, 'training_args.bin')
        torch.save(args, output_args_file)
    model.to(device)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

            model = Bert_CRF.from_pretrained(args.output_dir, bio_num_labels=bio_num_labels, type_num_labels=type_num_labels)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if(args.do_test and args.local_rank in [-1, 0]):
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

        model = Bert_CRF.from_pretrained(args.output_dir, bio_num_labels=bio_num_labels, type_num_labels=type_num_labels)

        model.to(args.device)
        test(args, model, tokenizer)


    return results




if __name__ == "__main__":
    main()
