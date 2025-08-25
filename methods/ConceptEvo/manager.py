import torch
import torch.nn.functional as F
import logging
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import BertTokenizer
from .evolution import concepts_init, concepts_evolution
from google import genai
from .losses import SupConLoss

__all__ = ['MULT']


def compute_discriminability_score(correct_scores, correct_labels, num_classes):
    """
    计算每个 concept 的判别性得分 D(c)

    参数:
        correct_scores: Tensor [M, concept_num]，预测正确样本的 concept 分数
        correct_labels: Tensor [M]，预测正确样本的标签
        correct_probs: Tensor [M]，预测正确样本的 softmax 最大值（置信度）
        num_classes: int，总类别数

    返回:
        D_c: Tensor [concept_num]，每个 concept 的判别性得分
        sim_y_c: Tensor [num_classes, concept_num]，原始响应
        sim_y_given_c: Tensor [num_classes, concept_num]，归一化后的条件概率
    """

    correct_scores = torch.abs(correct_scores)
    # concept_num = correct_scores.shape[1]
    class_concept_mean = {}

    # Step 1: 按类别加权平均 concept scores
    for i in range(num_classes):
        cls_mask = (correct_labels == i)
        if cls_mask.sum() > 0:
            mean_score = correct_scores[cls_mask].mean(dim=0)
            class_concept_mean[i] = mean_score
        else:
            class_concept_mean[i] = torch.zeros_like(correct_scores[0])

    # Step 2: 构造 Sim(y, c) 矩阵
    sim_y_c = torch.stack([class_concept_mean[i] for i in range(num_classes)])  # [Y, C]

    # Step 3: 归一化 Sim(y|c)
    sim_per_c = sim_y_c.mean(dim=0)  # [1, concept_num]

    sim_y_given_c = sim_per_c / (sim_per_c.sum() + 1e-6)
    # sim_y_given_c = F.softmax(sim_per_c, dim=0)

    # Step 4: 计算 D(c) = sum_y Sim(y|c) * log(Sim(y|c))
    D_c = (sim_y_given_c * torch.log(sim_y_given_c + 1e-6))  # [concept_num]

    min_d, max_d = D_c.min(), D_c.max()
    D_c = (D_c - min_d) / (max_d - min_d + 1e-5)

    return D_c

class MULT:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        
        self.device, self.model = model.device, model.model

        self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=args.wait_patience)

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']
        
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.contrast_criterion = SupConLoss()
        self.metrics = Metrics(args)

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_base_uncased_path, do_lower_case=True)
        args.genai_api_key = "sk-****************************"  # Please add your own API key here
        self.client = genai.Client(api_key=args.genai_api_key)

        self.key_concepts, self.history = concepts_init(self.client)


        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path)

        

    def _train(self, args): 

        early_stopping = EarlyStopping(args)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            contrastive_loss_record = AverageMeter()
            concepts_inputs = self.tokenizer(self.key_concepts, padding=True, truncation=True, return_tensors="pt")
            total_labels = torch.empty(0,dtype=torch.long).to(self.device)
            total_preds = torch.empty(0,dtype=torch.long).to(self.device)
            total_concept_scores = torch.empty((0, len(self.key_concepts))).to(self.device)
            total_logits = torch.empty((0, args.num_labels)).to(self.device)
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)  # [16,3,30]
                video_feats = batch['video_feats'].to(self.device)  # [16,230,256]
                audio_feats = batch['audio_feats'].to(self.device)  # [16,480,768]
                label_ids = batch['label_ids'].to(self.device)  # [16]

                with torch.set_grad_enabled(True):
                    batch_size = text_feats.shape[0]

                    all_logits, last_hiddens, concept_scores, avg_sim_score = self.model(text_feats, video_feats, audio_feats, concepts_inputs, contrast=True)
                    logits = all_logits[:batch_size]

                    total_logits = torch.cat((total_logits, logits))
                    total_labels = torch.cat((total_labels, label_ids))
                    total_concept_scores = torch.cat((total_concept_scores, concept_scores))

                    cls_loss = self.criterion(logits, label_ids)
                    loss = cls_loss
                    

                    self.optimizer.zero_grad()
                    
                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    self.optimizer.step()
            
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)
            y_score = total_concept_scores.detach().cpu() 
            y_pred = total_preds.cpu()
            y_true = total_labels.cpu()

            correct_mask = (y_pred == y_true)  # shape [N]
            correct_scores = y_score[correct_mask] # [M, concept_num]
            correct_labels = y_true[correct_mask]    # [M]
            num_classes = args.num_labels

            S_score = avg_sim_score.detach().cpu()  # [concept_num]

            D_score = compute_discriminability_score(correct_scores, correct_labels, num_classes)

            outputs = self._get_outputs(args, mode = 'eval')
            self.scheduler.step(outputs['loss'])
            eval_score = outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4),
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            early_stopping(eval_score, self.model, self.key_concepts)
            self.logger.info(f"Epoch {epoch + 1} concepts: {self.key_concepts}")
            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

            self.key_concepts, self.history = concepts_evolution(self.client, self.history, self.key_concepts, D_score, S_score, epoch+1, round(loss_record.avg, 4), round(eval_score, 4))

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model   
        self.key_concepts = early_stopping.concept
        
        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)   

    def _get_outputs(self, args, mode = 'eval', return_sample_results = False, show_results = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0, args.hidden_dim)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        loss_record = AverageMeter()

        concepts_inputs = self.tokenizer(self.key_concepts, padding=True, truncation=True, return_tensors="pt")

        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                
                logits, last_hiddens, _, _ = self.model(text_feats, video_feats, audio_feats, concepts_inputs, contrast=False)

                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, last_hiddens))
                total_labels = torch.cat((total_labels, label_ids))

                loss = self.criterion(logits, label_ids)
                loss_record.update(loss.item(), label_ids.size(0))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_feat = total_features.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        outputs.update({'loss': loss_record.avg})

        if return_sample_results:

            outputs.update(
                {
                    'y_feat': y_feat,
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            )

        return outputs

    
    def _test(self, args):
        test_results = self._get_outputs(args, mode = 'test', return_sample_results = True, show_results = True)
        test_results['best_eval_score'] = round(self.best_eval_score, 4)
    
        return test_results