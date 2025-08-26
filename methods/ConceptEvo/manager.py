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
from .evolution_ConceptEvo import concepts_init, concepts_evolution
from google import genai
from .losses import SupConLoss

__all__ = ['ConceptEvo']


def compute_discriminability_score(correct_scores, correct_labels, num_classes):
    """
    Compute the discriminative score D(c) for each concept.

    Args:
        correct_scores (Tensor): Shape [M, concept_num]. Concept scores for correctly
            predicted samples.
        correct_labels (Tensor): Shape [M]. Ground-truth labels of correctly
            predicted samples.
        correct_probs (Tensor): Shape [M]. Maximum softmax probabilities
            (confidence) of correctly predicted samples.
        num_classes (int): Total number of classes.

    Returns:
        D_c (Tensor): Shape [concept_num]. Discriminative score for each concept.
        sim_y_c (Tensor): Shape [num_classes, concept_num]. Raw responses.
        sim_y_given_c (Tensor): Shape [num_classes, concept_num]. Normalized
            conditional probabilities.
    """


    correct_scores = torch.abs(correct_scores)
    concept_num = correct_scores.shape[1]
    class_concept_mean = {}

    # Step 1: Weighted average concept scores by category
    for i in range(num_classes):
        cls_mask = (correct_labels == i)
        if cls_mask.sum() > 0:
            mean_score = correct_scores[cls_mask].mean(dim=0)
            class_concept_mean[i] = mean_score
        else:
            class_concept_mean[i] = torch.zeros_like(correct_scores[0])

    # Step 2: Construct Sim(y, c) matrix
    sim_y_c = torch.stack([class_concept_mean[i] for i in range(num_classes)])  # [Y, C]

    # Step 3: Normalized Sim(y|c)
    sim_per_c = sim_y_c.mean(dim=0)  # [1, concept_num]

    sim_y_given_c = sim_per_c / (sim_per_c.sum() + 1e-6)
    # sim_y_given_c = F.softmax(sim_per_c, dim=0)

    # Step 4: Calculate D(c) = sum_y Sim(y|c) * log(Sim(y|c))
    D_c = (sim_y_given_c * torch.log(sim_y_given_c + 1e-6))  # [concept_num]

    min_d, max_d = D_c.min(), D_c.max()
    D_c = (D_c - min_d) / (max_d - min_d + 1e-5)

    return D_c

class ConceptEvo:
    """
    Manager class for training, evaluation, and testing of the ConceptEvo model.
    """

    def __init__(self, args, data, model):

        # Logger for training information
        self.logger = logging.getLogger(args.logger_name)
        
        # Device and model reference
        self.device, self.model = model.device, model.model

        # Optimizer and LR scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=args.wait_patience
        )

        # Data loaders for train / dev / test
        self.train_dataloader = data.mm_dataloader['train']
        self.eval_dataloader = data.mm_dataloader['dev']
        self.test_dataloader = data.mm_dataloader['test']
        
        # Args and loss functions
        self.args = args
        self.criterion = nn.CrossEntropyLoss()     # classification loss
        self.contrast_criterion = SupConLoss()     # supervised contrastive loss
        self.metrics = Metrics(args)               # evaluation metrics

        # Tokenizer for concepts (used to encode key_concepts text into input_ids etc.)
        self.tokenizer = BertTokenizer.from_pretrained(
            args.bert_base_uncased_path, do_lower_case=True
        )

        # Gemini-2.0 client (for concept initialization and evolution)
        args.genai_api_key = "your_api_key"
        self.client = genai.Client(api_key=args.genai_api_key)

        # Initialize concept set and history via Gemini-2.0
        self.key_concepts, self.history = concepts_init(self.client)

        # Initialize best evaluation score
        if args.train:
            self.best_eval_score = 0
        else:
            # Load trained model from disk if not training
            self.model = restore_model(self.model, args.model_output_path)

    # Training
    def _train(self, args): 
        """
        Training loop with concept evolution.
        Each epoch:
          1. Train the model on batches (classification + contrastive losses).
          2. Collect prediction results to compute concept discriminability/diversity scores.
          3. Use Gemini-2.0 to evolve concepts based on feedback (D_score + S_score).
          4. Apply early stopping based on validation results.
        """

        early_stopping = EarlyStopping(args)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            contrastive_loss_record = AverageMeter()

            # Encode current concept set as input for the model
            concepts_inputs = self.tokenizer(
                self.key_concepts, padding=True, truncation=True, return_tensors="pt"
            )

            # Containers for epoch statistics
            total_labels = torch.empty(0, dtype=torch.long).to(self.device)
            total_preds = torch.empty(0, dtype=torch.long).to(self.device)
            total_concept_scores = torch.empty((0, len(self.key_concepts))).to(self.device)
            total_logits = torch.empty((0, args.num_labels)).to(self.device)
            
            # Iterate over training batches
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                # Extract multimodal features and labels
                text_feats = batch['text_feats'].to(self.device)    # [B, T_text, D_text]
                video_feats = batch['video_feats'].to(self.device)  # [B, T_video, D_video]
                audio_feats = batch['audio_feats'].to(self.device)  # [B, T_audio, D_audio]
                label_ids = batch['label_ids'].to(self.device)      # [B]

                with torch.set_grad_enabled(True):
                    batch_size = text_feats.shape[0]

                    # Forward pass with contrastive view (contrast=True)
                    all_logits, last_hiddens, concept_scores, avg_sim_score, contrast_logits = \
                        self.model(text_feats, video_feats, audio_feats, concepts_inputs, contrast=True)

                    # First half of logits are normal samples
                    logits = all_logits[:batch_size]

                    # Collect outputs for later statistics
                    total_logits = torch.cat((total_logits, logits))
                    total_labels = torch.cat((total_labels, label_ids))
                    total_concept_scores = torch.cat((total_concept_scores, concept_scores))

                    # Classification loss
                    cls_loss = self.criterion(logits, label_ids)

                    # Uniform loss on perturbed (masked) logits
                    perturbed_logits = all_logits[batch_size:]  # [B, num_classes]
                    num_classes = perturbed_logits.size(1)
                    uniform_targets = torch.full_like(perturbed_logits, 1.0 / num_classes)
                    log_probs = torch.log_softmax(perturbed_logits, dim=1)
                    uniform_loss = -(uniform_targets * log_probs).sum(dim=1).mean()

                    # Contrastive loss
                    # Re-run forward to generate augmented view
                    _, _, _, _, contrast_logits_aug = self.model(
                        text_feats, video_feats, audio_feats, concepts_inputs, contrast=True
                    )
                    norm_logits = F.normalize(contrast_logits)
                    norm_logits_aug = F.normalize(contrast_logits_aug)

                    # Build mask for positive/negative pairs
                    labels_aug = torch.full((batch_size,), args.num_labels + 1).to(self.device)
                    all_labels = torch.cat((label_ids, labels_aug))
                    labels_expand = all_labels.expand(2 * batch_size, 2 * batch_size)
                    mask = torch.eq(labels_expand, labels_expand.T).long()
                    mask[all_labels == args.num_labels + 1, :] = 0  # remove augmented noise labels

                    logits_mask = torch.scatter(
                        mask,
                        0,
                        torch.arange(2 * batch_size).unsqueeze(0).to(self.device),
                        1
                    )

                    # Prepare logits for SupConLoss
                    contrastive_logits = torch.cat(
                        (norm_logits.unsqueeze(1), norm_logits_aug.unsqueeze(1)), dim=1
                    )
                    
                    contrastive_loss = self.contrast_criterion(
                        contrastive_logits, mask=logits_mask,
                        temperature=args.temperature, device=self.device
                    )

                    # Final loss = classification + contrastive
                    loss = cls_loss + contrastive_loss

                    # Backprop
                    self.optimizer.zero_grad()
                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    contrastive_loss_record.update(contrastive_loss.item(), label_ids.size(0))

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(
                            [param for param in self.model.parameters() if param.requires_grad],
                            args.grad_clip
                        )
                    self.optimizer.step()
            
            # After each epoch: evaluate + concept evolution 
            # Gather predictions
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim=1)
            y_score = total_concept_scores.detach().cpu() 
            y_pred = total_preds.cpu()
            y_true = total_labels.cpu()

            # Select correctly predicted samples for D_score
            correct_mask = (y_pred == y_true)
            correct_scores = y_score[correct_mask]    # [M, concept_num]
            correct_labels = y_true[correct_mask]     # [M]
            num_classes = args.num_labels

            # Concept diversity & discriminability
            S_score = avg_sim_score.detach().cpu()  # [concept_num]
            D_score = compute_discriminability_score(correct_scores, correct_labels, num_classes)

            # Validation evaluation
            outputs = self._get_outputs(args, mode='eval')
            self.scheduler.step(outputs['loss'])
            eval_score = outputs[args.eval_monitor]

            # Log results
            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'contrastive_loss': round(contrastive_loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4),
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            # Early stopping check
            early_stopping(eval_score, self.model, self.key_concepts)
            self.logger.info(f"Epoch {epoch + 1} concepts: {self.key_concepts}")
            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

            # Concept evolution via Gemini-2.0 
            result = concepts_evolution(
                self.client, self.history, self.key_concepts,
                D_score, S_score, epoch+1,
                round(loss_record.avg, 4), round(eval_score, 4)
            )
            if len(result) == 3:
                self.key_concepts, self.history, concept_info = result
                self.logger.info(f"Epoch {epoch + 1} concepts: {concept_info}")
            else:
                self.key_concepts, self.history = result
                self.logger.warning(f"Epoch {epoch + 1} concepts: concept_info not returned.")

        # Save best model + concepts
        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model   
        self.key_concepts = early_stopping.concept
        
        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)   

    # Evaluation
    def _get_outputs(self, args, mode='eval', return_sample_results=False, show_results=False):
        """
        Run model in eval/test mode on the given dataloader.
        Returns loss, predictions, metrics, and optionally raw features.
        """
        
        # Select dataloader
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        # Containers
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        total_features = torch.empty((0, args.hidden_dim)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        loss_record = AverageMeter()

        # Encode current concept set
        concepts_inputs = self.tokenizer(
            self.key_concepts, padding=True, truncation=True, return_tensors="pt"
        )

        # Iterate over batches
        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                logits, last_hiddens, _, _, _ = self.model(
                    text_feats, video_feats, audio_feats, concepts_inputs, contrast=False
                )

                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, last_hiddens))
                total_labels = torch.cat((total_labels, label_ids))

                loss = self.criterion(logits, label_ids)
                loss_record.update(loss.item(), label_ids.size(0))

        # Predictions
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim=1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_feat = total_features.cpu().numpy()

        # Compute metrics
        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        outputs.update({'loss': loss_record.avg})

        # Optionally return raw results
        if return_sample_results:
            outputs.update({
                'y_feat': y_feat,
                'y_true': y_true,
                'y_pred': y_pred
            })

        return outputs
    # Testing
    def _test(self, args):
        """
        Run testing with the best model and return results + sample outputs.
        """
        test_results = self._get_outputs(
            args, mode='test', return_sample_results=True, show_results=True
        )
        test_results['best_eval_score'] = round(self.best_eval_score, 4)
    
        return test_results
