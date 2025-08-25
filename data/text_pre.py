import os
import csv
import sys
import logging
from transformers import BertTokenizer

__all__ = ['TextDataset']

class TextDataset:
    
    def __init__(self, args, base_attrs):
        
        self.logger = logging.getLogger(args.logger_name)
        self.base_attrs = base_attrs
        
        if args.text_backbone.startswith('bert'):
            self.feats = self._get_feats(args, base_attrs)
        else:
            raise Exception('Error: inputs are not supported text backbones.')

    def _get_feats(self, args, base_attrs):

        self.logger.info('Generate Text Features Begin...')

        processor = DatasetProcessor(args)

        train_examples = processor.get_examples(base_attrs['data_path'], 'train')
        train_feats = self._get_bert_feats(args, train_examples, base_attrs)

        dev_examples = processor.get_examples(base_attrs['data_path'], 'dev')
        dev_feats = self._get_bert_feats(args, dev_examples, base_attrs)

        test_examples = processor.get_examples(base_attrs['data_path'], 'test')
        test_feats = self._get_bert_feats(args, test_examples, base_attrs)

        if args.aug:
            aug_examples = processor.get_examples(base_attrs['data_path'], 'aug')
            aug_feats = self._get_bert_feats(args, aug_examples, base_attrs)

        self.logger.info('Generate Text Features Finished...')

        train_text_feats, dev_text_feats, test_text_feats = train_feats['features'], dev_feats['features'], test_feats['features']
        text_feats = {
            'train': train_text_feats,
            'dev': dev_text_feats,
            'test': test_text_feats
        }
        outputs = {'text_feats': text_feats}

        if args.method == 'tcl_map':
            train_cons_text_feats, dev_cons_text_feats, test_cons_text_feats = train_feats['cons_features'], dev_feats['cons_features'], test_feats['cons_features']
            train_condition_idx, dev_condition_idx, test_condition_idx = train_feats['condition_idx'], dev_feats['condition_idx'], test_feats['condition_idx']
            cons_text_feats = {'train': train_cons_text_feats, 'dev': dev_cons_text_feats, 'test': test_cons_text_feats}
            condition_idx = { 'train': train_condition_idx, 'dev': dev_condition_idx, 'test': test_condition_idx}
            outputs['cons_text_feats'], outputs['condition_idx'] = cons_text_feats, condition_idx

        if args.method == 'sdif':
            if args.aug:
                aug_text_feats = aug_feats['features']
                text_feats = {'train': train_text_feats, 'dev': dev_text_feats, 'test': test_text_feats, 'aug': aug_text_feats}
                outputs = {'text_feats': text_feats}            
        
        return outputs    

    def _get_bert_feats(self, args, examples, base_attrs):

        if args.text_backbone.startswith('bert'):
            tokenizer = BertTokenizer.from_pretrained(args.bert_base_uncased_path, do_lower_case=True)

        if args.method == 'tcl_map':
            features, cons_features, condition_idx, args.max_cons_seq_length = convert_examples_to_features_tcl_map(args, examples, base_attrs, tokenizer)     
            features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]
            cons_features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in cons_features]
            outputs = {
                'features': features_list,
                'cons_features': cons_features_list,
                'condition_idx': condition_idx,
            }
            return outputs
        
        features = convert_examples_to_features(examples, base_attrs, tokenizer)     
        features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]
        outputs = {
            'features': features_list,
        }
        return outputs

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def __init__(self, args):
        super(DatasetProcessor).__init__()

        if args.dataset in ['MIntRec']:
            self.text_id = 3
            self.label_id = 4
        elif args.dataset in ['MIntRec2.0']:
            self.text_id = 2
            self.label_id = 3
        elif args.dataset in ['MELD-DA']:
            self.text_id = 2
            self.label_id = 3
        elif args.dataset in ['IEMOCAP-DA']:
            self.text_id = 1
            self.label_id = 2          

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'dev':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        elif mode == 'aug':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "augment_train.tsv")), "aug")
        
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[self.text_id]
            label = line[self.label_id]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, base_attrs, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    max_seq_length = base_attrs["benchmarks"]['max_seq_lengths']['text']
    
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
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

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    return features


def convert_examples_to_features_tcl_map(args, examples, base_attrs, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    base_attrs['prompt_len'] = args.prompt_len  # 3
    base_attrs['label_len'] = args.label_len  # 4
    
    max_seq_length = base_attrs['benchmarks']['max_seq_lengths']['text']  # 30
    label_len = base_attrs['label_len']
    features = []
    cons_features = []
    condition_idx = []
    prefix = ['MASK'] * base_attrs['prompt_len']  # ['MASK', 'MASK', 'MASK']
    max_cons_seq_length = max_seq_length + len(prefix) + label_len  # 37=30+3+4

    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        
        # if args.dataset in ['MIntRec']:
        #     condition = tokenizer.tokenize(example.label)
        # elif args.dataset in ['MELD']:
        #     condition = tokenizer.tokenize(base_attrs['bm']['label_maps'][example.label])
        condition = tokenizer.tokenize(example.label)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # construct augmented sample pair
        cons_tokens = ["[CLS]"] + tokens_a + prefix + condition + (label_len - len(condition)) * ["MASK"] + ["[SEP]"]
        tokens = ["[CLS]"] + tokens_a + prefix + label_len * ["[MASK]"] + ["[SEP]"]

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        cons_inputs_ids = tokenizer.convert_tokens_to_ids(cons_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_cons_seq_length - len(input_ids))
        input_ids += padding
        cons_inputs_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_cons_seq_length
        assert len(cons_inputs_ids) == max_cons_seq_length
        assert len(input_mask) == max_cons_seq_length
        assert len(segment_ids) == max_cons_seq_length
        # record the position of prompt
        condition_idx.append(1 + len(tokens_a) + len(prefix))


        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
        
        cons_features.append(
            InputFeatures(input_ids=cons_inputs_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    
    return features, cons_features, condition_idx, max_cons_seq_length

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()