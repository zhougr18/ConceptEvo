import os
import logging
import csv
from torch.utils.data import DataLoader

from .mm_pre import MMDataset
from .text_pre import TextDataset
from .video_pre import VideoDataset
from .audio_pre import AudioDataset
from .mm_pre import MMDataset, AuGDataset
from .__init__ import benchmarks

__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args, logger_name = 'Multimodal Intent Recognition'):
        
        self.logger = logging.getLogger(logger_name)
        self.data_path = os.path.join(args.data_path, args.dataset)
        self.benchmarks = benchmarks[args.dataset]
        self.label_list = self.benchmarks["intent_labels"]
        self.logger.info('Lists of intent labels are: %s', str(self.label_list))
        args.num_labels = len(self.label_list)
        args.text_seq_len, args.video_seq_len, args.audio_seq_len = \
            self.benchmarks['max_seq_lengths']['text'], self.benchmarks['max_seq_lengths']['video'], self.benchmarks['max_seq_lengths']['audio']

        self.train_data_index, self.train_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'train.tsv'), args)
        self.dev_data_index, self.dev_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'dev.tsv'), args)
        self.test_data_index, self.test_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'test.tsv'), args)
        args.num_train_examples = len(self.train_data_index)
        if args.aug:
            self.aug_data_index, self.aug_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'augment_train.tsv'), args)
        
        self.unimodal_feats = self._get_unimodal_feats(args, self._get_attrs())
        self.mm_data = self._get_multimodal_data(args)
        self.mm_dataloader = self._get_dataloader(args, self.mm_data)
    
    def _get_unimodal_feats(self, args, attrs):
        
        text_feats = TextDataset(args, attrs).feats
        video_feats = VideoDataset(args, attrs).feats
        audio_feats = AudioDataset(args, attrs).feats

        return {
            'text': text_feats,
            'video': video_feats,
            'audio': audio_feats
        }
    
    def _get_multimodal_data(self, args):

        text_data = self.unimodal_feats['text']['text_feats']
        video_data = self.unimodal_feats['video']
        audio_data = self.unimodal_feats['audio']
        
        text = self.unimodal_feats['text']
        other_data = self._get_other_data(text)

        mm_train_data = MMDataset(self.train_label_ids, text_data['train'], video_data['train'], audio_data['train'], other_data['train'])
        mm_dev_data = MMDataset(self.dev_label_ids, text_data['dev'], video_data['dev'], audio_data['dev'], other_data['dev'])
        mm_test_data = MMDataset(self.test_label_ids, text_data['test'], video_data['test'], audio_data['test'], other_data['test'])

        if args.aug:
            mm_aug_data = AuGDataset(self.aug_label_ids, text_data['aug'])
            return {
                'train': mm_train_data,
                'aug': mm_aug_data,
                'dev': mm_dev_data,
                'test': mm_test_data
            }
        
        return {
            'train': mm_train_data,
            'dev': mm_dev_data,
            'test': mm_test_data
        }

    def _get_dataloader(self, args, data):
        
        self.logger.info('Generate Dataloader Begin...')

        train_dataloader = DataLoader(data['train'], shuffle=True, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
        dev_dataloader = DataLoader(data['dev'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        test_dataloader = DataLoader(data['test'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        
        self.logger.info('Generate Dataloader Finished...')

        if args.aug:
            aug_dataloader = DataLoader(data['aug'], shuffle=True, batch_size = args.aug_batch_size)
            return {
                'train': train_dataloader,
                'aug': aug_dataloader,
                'dev': dev_dataloader,
                'test': test_dataloader
            }
                    
        return {
            'train': train_dataloader,
            'dev': dev_dataloader,
            'test': test_dataloader
        }
        
    def _get_attrs(self):

        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs
    
    def _get_other_data(self, inputs):
        other_data = {}
        other_data['train'] = {}
        other_data['dev'] = {}
        other_data['test'] = {}

        for key in inputs.keys():
            if key not in ['text_feats']:
                if 'train' in inputs[key]:
                    other_data['train'][key] = inputs[key]['train']
                if 'dev' in inputs[key]:
                    other_data['dev'][key] = inputs[key]['dev']
                if 'test' in inputs[key]:
                    other_data['test'][key] = inputs[key]['test']

        return other_data
            
    def _get_indexes_annotations(self, read_file_path, args):

        label_map = {}
        for i, label in enumerate(self.label_list):
            label_map[label] = i

        with open(read_file_path, 'r') as f:

            data = csv.reader(f, delimiter="\t")
            indexes = []
            label_ids = []

            for i, line in enumerate(data):
                if i == 0:
                    continue
                
                if args.dataset in ['MIntRec']:
                    index = '_'.join([line[0], line[1], line[2]])                
                    label_id = label_map[line[4]]          
                
                elif args.dataset in ['MIntRec2.0']:
                    index ='_'.join(['dia' + str(line[0]), 'utt' + str(line[1])])   
                    label_id = label_map[line[3]]

                elif args.dataset in ['MELD-DA']:
                    index = '_'.join([line[0], line[1]])
                    label_id = label_map[line[3]]
                
                elif args.dataset in ['IEMOCAP-DA']:
                    index = line[0]
                    label_id = label_map[line[2]]
                
                indexes.append(index)
                label_ids.append(label_id)

        return indexes, label_ids