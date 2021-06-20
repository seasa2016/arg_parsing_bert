import json
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        raise ValueError('the length should larger than all max {0} now {1}'.format(max_length, total_length))
        tokens_b.pop()

def convert_examples_to_features(examples, tokenizer,
                  max_seq_length=128,
                  label_list=None, output_mode=None,
                  pad_on_left=False,
                  pad_token=0,
                  pad_token_segment_id=0,
                  mask_padding_with_zero=True, train_eval=True):
    """
    Loads a data file into a list of `InputBatch`s
    """
    logger.info("Using label list %s" % (label_list))
    logger.info("Using output mode %s" % (output_mode))

    label_map =[
        {label : i for i, label in enumerate(label_list[0])},
        {label : i for i, label in enumerate(label_list[1])}
    ]
    label_map[1]['N'] = -1

    features = []
    print('length:', len(examples))
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example['context'])
        #print(labels)

        tokens_b = None
        if(example['topic']):
            tokens_b = tokenizer.tokenize(example['topic'])
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            try:
                truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            except:
                continue
                print(example)
                print(len(tokens_a), tokens_a)
                print(len(tokens_b), tokens_b)
        else:
            None+1
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The attention mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # prepare recover array
        recover = []
        now, index = 0, 0
        for word in example['context'].split():
            length = len(tokenizer.tokenize(word))
            recover.extend( list(range(length)) )


        input_mask_a = [1] * len(recover)
        input_mask_a.extend( [0] * (max_seq_length - len(input_mask_a)) )
        recover.extend([2]*(max_seq_length-len(recover)))


        # prepare label for bio setting
        bio_label_id = []
        type_label_id = []

        if(output_mode == "classification" and train_eval):
            index, type_index  = 0, 0

            dtype = example['type'].split()

            for word, bio in zip(example['context'].split(), example['bio'].split()):
                l = len(tokenizer.tokenize(word))

                if(bio == 'O'):
                    type_label_id.extend( [-1]*l )
                    bio_label_id.extend( [ label_map[0]['O'] ]*l )
                else:
                    type_label_id.extend( [label_map[1][ dtype[type_index] ]]*l )

                    if(bio == 'B'):
                        bio_label_id.append( label_map[0]['B'] )
                        l -= 1

                    bio_label_id.extend( [ label_map[0]['I'] ]*l )

                    if(bio == 'E'):
                        bio_label_id[-1] = label_map[0]['E']
                        type_index += 1

            assert(len(bio_label_id) == len(type_label_id)), 'length of bio and type should be the same'
            bio_label_id.extend([2]*(max_seq_length-len(bio_label_id)))
            type_label_id.extend([-1]*(max_seq_length-len(type_label_id)))

        elif(output_mode == "regression" and train_eval):
            raise TypeError('we use only classification')
        elif(train_eval == False):
            #print('you are in test mode')
            pass
        else:
            raise KeyError(output_mode)

        # since there will be extending words, we need to check the excat
        #######################################################################
        if(ex_index < 5):
            logger.info("*** Example ***")
            print(len(tokens_a), tokens_a)
            print(len(example['context']), example['context'])
            logger.info("guid: %s" % (example['uid']))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("recover_ids: %s" % " ".join([str(x) for x in recover]))
            if(train_eval):
                print(len(example['bio']), example['bio'])
                logger.info("bio label: %s" % (example['bio']))
                logger.info("bio_id : %s" % (' '.join([ str(_) for _ in bio_label_id])))
                logger.info("type label: %s" % (example['type']))
                logger.info("type_id : %s" % (' '.join([ str(_) for _ in type_label_id])))

        # ['input_ids', 'attention_mask', 'crf_mask', 'segment_ids', 'label_id', 'recover']
        if(train_eval):
            features.append({
                'input_ids':input_ids,
                'attention_mask':input_mask,
                'crf_mask':input_mask_a,
                'segment_ids':segment_ids,
                'bio_labels':bio_label_id,
                'type_labels':type_label_id,
                'recover':recover,
                'index':example['index']
                })
        else:
            features.append({
                'input_ids':input_ids,
                'attention_mask':input_mask,
                'crf_mask':input_mask_a,
                'segment_ids':segment_ids,
                'recover':recover,
                'index':example['index']
                })

    print('features length is ', len(features))
    return features

class ArgumentProcessor(object):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._read_jsons(data_dir+"_train", "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._read_jsons(data_dir+"_eval", "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._read_jsons(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return [["B", "I", "O", "E"], ['P', 'C']]

    def _read_jsons(self, filename, mode):
        datas = []
        index = 0
        with open(filename) as f:
            f.readline()
            for line in f:
                temp = json.loads(line)
                for index in range(len(temp['context'])):
                    if(temp['context'][index].strip()==''):
                        continue
                    datas.append({
                            'uid':'{0}_{1}'.format(mode, index),
                            'topic':temp['topic'],
                            'context':temp['context'][index],
                            'index':len(datas)
                        })
                    try:
                        datas[-1]['bio'] = temp['bio'][index]
                        datas[-1]['type'] = temp['label'][index]
                    except:
                        pass

                    try:
                        datas[-1]['uid'] = '{}_{}'.format(temp['uid'], index)
                    except:
                        pass
                    """
                    if(len(datas)==10000):
                        return datas
                    """
        if(mode == 'test'):
            with open('./mapping/train/mapping_{}'.format(filename.split('/')[-1]), 'w') as f:
            #with open('./mapping/heldout/mapping_{}'.format(filename.split('/')[-1]), 'w') as f:
            #with open('./mapping/v4/mapping_{}'.format(filename.split('/')[-1]), 'w') as f:
                for _ in datas:
                    f.write(_['uid'])
                    f.write('\n')
        return datas

if(__name__ == '__main__'):
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    processor = TreeProcessor()
    data = {}
    data['train'] = processor.get_train_examples('./../../preprocess/parsing/ArgumentEssays')
    data['dev'] = processor.get_dev_examples('./../../preprocess/parsing/ArgumentEssays')

    feat = {}
    feat['train'] = convert_examples_to_features(data['train'], tokenizer, logger,
                                        max_word_length=128, max_sent_length=32,
                                        label_list=processor.get_labels(), output_mode='classification',
                                        pad_on_left=False,
                                        pad_token=0,
                                        pad_token_segment_id=0,
                                        mask_padding_with_zero=True)

    feat['dev'] = convert_examples_to_features(data['dev'], tokenizer, logger,
                                        max_word_length=128, max_sent_length=32,
                                        label_list=processor.get_labels(), output_mode='classification',
                                        pad_on_left=False,
                                        pad_token=0,
                                        pad_token_segment_id=0,
                                        mask_padding_with_zero=True)

    for key in ['train', 'dev']:
        m = 0
        for _ in data['train']:
            m = max(m, len(_['sent']))
        print(key, 'sent', m)

    for key in ['train', 'dev']:
        m = 0
        for _ in feat[key]:
            for attn in _['word_attention_mask']:
                m = max(m, sum(attn))
        print(key, 'word_attention_mask', m)

    for key in ['train', 'dev']:
        m = 0
        mm = 0
        for _ in feat[key]:
            m = max(m, sum(_['sent_attention_mask']))
            mm = max(m, len(_['sent_attention_mask']))
        print(key, 'sent_attention_mask', m, mm)

    for key in feat['train'][0]:
        temp = [ _[key] for _ in feat['train']]
        print(key, torch.tensor(temp).shape)

    print('data processor pass')
