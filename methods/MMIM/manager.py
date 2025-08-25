import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm, trange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.metrics import AverageMeter, Metrics
from utils.functions import restore_model, save_model, EarlyStopping
from torch.utils.data import DataLoader

__all__ = ['MMIM']

class MMIM(nn.Module):
    
    def __init__(self, args, data, model):
        
        super(MMIM, self).__init__()
        self.logger = logging.getLogger(args.logger_name)
        self.device, self.model = model.device, model.model
        
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']

        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)

    def _train_mmilb(self, args):
        
        self.model.train()
        loss_record = AverageMeter()

        with tqdm(self.train_dataloader) as td:
            
            for batch in td:
                
                self.model.zero_grad()
                
                text_feats = batch['text_feats'].to(self.device)
                video_lengths = batch['video_lengths'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                video_data = {'feats': video_feats, 'lengths': video_lengths}
                
                audio_lengths = batch['audio_lengths'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                audio_data = {'feats': audio_feats, 'lengths': audio_lengths}
                label_ids = batch['label_ids'].to(self.device)

                results = self.model(text_feats, video_data, audio_data, mode = 'train')

                loss = -results['lld']
                loss.backward()
                loss_record.update(loss.item(), label_ids.size(0))

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                self.optimizer_mmilb.step()
            
        return loss_record.avg

    def _train_others(self, args):
        
        y_pred, y_true = [], []
        loss_record = AverageMeter()
            
        self.model.train()

        with tqdm(self.train_dataloader) as td:
            for i_batch, batch in enumerate(td):

                self.model.zero_grad()
                text_feats = batch['text_feats'].to(self.device)
                video_lengths = batch['video_lengths'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                video_data = {'feats': video_feats, 'lengths': video_lengths}
                
                audio_lengths = batch['audio_lengths'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                audio_data = {'feats': audio_feats, 'lengths': audio_lengths}
                label_ids = batch['label_ids'].to(self.device)

                results = self.model(text_feats, video_data, audio_data, mode = 'train')
                
                y_loss = self.criterion(results['M'], label_ids)

                if args.contrast:
                    loss = y_loss + args.alpha * results['nce'] - args.beta * results['lld']
                else:
                    loss = y_loss

                loss.backward()
                loss_record.update(loss.item(), label_ids.size(0))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                
                self.optimizer_main.step()
                
                if label_ids.shape[0] != 1:
                    label_ids = label_ids.squeeze()
                y_pred.append(results['M'].cpu())
                y_true.append(label_ids.cpu())
                
        pred, truth = torch.cat(y_pred), torch.cat(y_true)
        # pred, truth = None, None
        return loss_record.avg, pred, truth

    def _train(self, args):
        
        early_stopping = EarlyStopping(args)

        mmilb_param = []
        main_param = []
        bert_param = []

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if 'bert' in name:
                    bert_param.append(p)
                elif 'mi' in name:
                    mmilb_param.append(p)
                else: 
                    main_param.append(p)
            
            for p in (mmilb_param+main_param):
                if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                    nn.init.xavier_normal_(p)
        
        if len(mmilb_param) != 0:
            self.optimizer_mmilb = getattr(torch.optim, args.optim)(
                list(mmilb_param), lr = args.lr_mmilb, weight_decay=args.weight_decay_mmilb)
       
        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': args.weight_decay_bert, 'lr': args.lr_bert},
            {'params': main_param, 'weight_decay': args.weight_decay_main, 'lr': args.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, args.optim)(
            optimizer_main_group
        )

        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='max', patience=args.wait_patience, factor=0.5, verbose=True)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            if args.contrast:
                train_loss_mmilb = self._train_mmilb(args)
                
            train_loss_main, pred, truth = self._train_others(args)

            outputs = self._get_outputs(args, self.eval_dataloader)
            eval_score = outputs[args.eval_monitor]
            self.scheduler_main.step(eval_score)

            eval_results = {
                'train_loss_main': round(train_loss_main, 4),
                'train_loss_mmilb': round(train_loss_mmilb, 4),
                'eval_score': round(eval_score, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)

    def _get_outputs(self, args, dataloader, show_results = False):

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.feat_size)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_lengths = batch['video_lengths'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            video_data = {'feats': video_feats, 'lengths': video_lengths}
            
            audio_lengths = batch['audio_lengths'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            audio_data = {'feats': audio_feats, 'lengths': audio_lengths}
            label_ids = batch['label_ids'].to(self.device)
  
            with torch.set_grad_enabled(False):
                
                logits, features = self.model(text_feats, video_data, audio_data)
                
                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_logit = total_logits.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_prob = total_maxprobs.cpu().numpy()
        y_feat = total_features.cpu().numpy()
        
        outputs = self.metrics(y_true, y_pred, show_results = show_results)
            
        outputs.update(
            {
                'y_prob': y_prob,
                'y_logit': y_logit,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_feat': y_feat
            }
        )

        return outputs

    def _test(self, args):
        
        test_results = {}
        
        ind_test_results = self._get_outputs(args, self.test_dataloader, show_results = True)
        if args.train:
            test_results['best_eval_score'] = round(self.best_eval_score, 4)
        test_results.update(ind_test_results)        
        
        return test_results