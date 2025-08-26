# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

from ..SubNets.FeatureNets import BERTEncoder
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from ..SubNets.AlignNets import AlignSubNet

__all__ = ['ConceptEvo_GPT']


def get_transformer_encoder(args, embed_dim: int, layers: int) -> TransformerEncoder:
    """
    To build a TransformerEncoder with settings from args.

    Args:
        args: nheads, attn_dropout, relu_dropout, res_dropout, embed_dropout, attn_mask
        embed_dim: hidden size of the encoder
        layers: number of layers/blocks
    """
    return TransformerEncoder(
        embed_dim=embed_dim,
        num_heads=args.nheads,
        layers=layers,
        attn_dropout=args.attn_dropout,
        relu_dropout=args.relu_dropout,
        res_dropout=args.res_dropout,
        embed_dropout=args.embed_dropout,
        attn_mask=args.attn_mask,
    )


class MultimodalFusion(nn.Module):
    """
    Cross-modal fusion block.

    Pipeline:
        1) Project raw features of V/A/L into a common hidden dim D.
        2) Cross-attend L with V and A respectively to get L<-V and L<-A views.
        3) Concatenate [L, L<-V, L<-A] along channel dim and fuse via Linear+ReLU+Dropout.
        4) Optional final Transformer encoding for deeper fusion.
        
    """

    def __init__(self, args):
        super(MultimodalFusion, self).__init__()

        # Linear projections to a shared hidden space D
        self.v_proj = nn.Linear(args.video_feat_dim, args.hidden_dim)
        self.a_proj = nn.Linear(args.audio_feat_dim, args.hidden_dim)
        self.l_proj = nn.Linear(args.text_feat_dim, args.hidden_dim)

        # Temporal/sequence alignment
        self.align_net = AlignSubNet(args, args.aligned_method)

        # Cross-attention encoders
        self.cross_attn_vl = get_transformer_encoder(args, args.hidden_dim, args.vl_layers)  # L <- V
        self.cross_attn_al = get_transformer_encoder(args, args.hidden_dim, args.al_layers)  # L <- A

        # Fuse [L, L<-V, L<-A] -> D
        self.fusion = nn.Linear(3 * args.hidden_dim, args.hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(args.fusion_dropout)

        # Final Transformer encoder on fused sequence
        self.final_encoder = get_transformer_encoder(args, args.hidden_dim, args.fusion_layers)

    def forward(self, x_l: torch.Tensor, x_v: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_l: [B, T, D_l] text features
            x_v: [B, T, D_v] video features 
            x_a: [B, T, D_a] audio features

        Returns:
            fusion_feats: [B, T, D] fused representation
        """

        # AlignSubNet expects [T, B, C]; we permute, align, then permute back.
        x_l, x_a, x_v = self.align_net(
            x_l.permute(1, 0, 2),  # [T, B, D_l]
            x_a.permute(1, 0, 2),  # [T, B, D_a]
            x_v.permute(1, 0, 2),  # [T, B, D_v]
        )
        x_l, x_a, x_v = x_l.permute(1, 0, 2), x_a.permute(1, 0, 2), x_v.permute(1, 0, 2)  # -> [B, T, *]

        # Project to common hidden size D
        x_l = self.l_proj(x_l)  # [B, T, D]
        x_v = self.v_proj(x_v)  # [B, T, D]
        x_a = self.a_proj(x_a)  # [B, T, D]

        # Cross-attention
        x_vl = self.cross_attn_vl(x_l, x_v, x_v)  # [B, T, D], L attended to V
        x_al = self.cross_attn_al(x_l, x_a, x_a)  # [B, T, D], L attended to A

        # Concatenate three branches then fuse
        fusion_feats = torch.cat([x_l, x_vl, x_al], dim=-1)  # [B, T, 3D]
        fusion_feats = self.fusion(fusion_feats)             # [B, T, D]
        fusion_feats = self.activation(fusion_feats)
        fusion_feats = self.dropout(fusion_feats)

        # Deeper fusion via Transformer
        fusion_feats = self.final_encoder(fusion_feats)      # [B, T, D]
        return fusion_feats


class ConceptEvo_GPT(nn.Module):
    """
    Concept-guided multimodal model.

    High-level:
        - Text encoder: BERTEncoder (CLS + token features).
        - Video/Audio encoders: shallow TransformerEncoders over temporal sequences.
        - Concept bank (text prompts) -> BERT -> projected to D, used to compute
          token-to-concept similarities and gate fused features.
        - Fusion: MultimodalFusion (L with cross-attn from V/A) + gate with concept features.
        - Heads: classification (out_layer), contrastive (contrast_head).
        
    """

    def __init__(self, args):
        super(ConceptEvo_GPT, self).__init__()

        # Text encoder
        self.text_subnet = BERTEncoder.from_pretrained(args.bert_base_uncased_path)

        # Dims & hyperparams
        self.video_feat_dim = args.video_feat_dim    # e.g., 256
        self.text_feat_dim = args.text_feat_dim      # e.g., 768
        self.audio_feat_dim = args.audio_feat_dim    # e.g., 768
        self.hidden_dim = args.hidden_dim            # e.g., 120
        self.mask_ratio = args.mask_ratio            # e.g., 0.2

        # Modal encoders
        self.v_encoder = get_transformer_encoder(args, args.video_feat_dim, args.encoder_layers_v)
        self.a_encoder = get_transformer_encoder(args, self.audio_feat_dim, args.encoder_layers_a)

        # Concept projection
        self.concept_proj = nn.Linear(self.text_feat_dim, self.hidden_dim)  # 768 -> D
        self.relu = nn.ReLU()
        self.dropout_c = nn.Dropout(p=args.concept_dropout)  # concept dropout

        # Fusion & gate
        self.fusion_net = MultimodalFusion(args)
        self.fusion_gate_fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)  # gate on [fusion, concept-aug]

        # Heads
        # Light MLP before heads (token-average -> utterance-level)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=args.fc_dropout_1),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=args.fc_dropout_2),
        )

        self.contrast_head = nn.Linear(self.hidden_dim, args.num_labels)  # for contrastive logits if needed
        self.out_layer = nn.Linear(self.hidden_dim, args.num_labels)      # classifier logits

    # Concept features

    def get_concept_feats(self, concepts_inputs: dict, device: torch.device):
        """
        Encode the concept texts, compute their mutual similarity, and return:
            - concepts_feats (projected to D)
            - avg_sim_score (normalized average pairwise similarity per concept)

        Args:
            concepts_inputs: dict with keys: input_ids, attention_mask, token_type_ids
                             Each is [C, Lc] where C=#concepts, Lc=concept token length
            device: torch device to place tensors on

        Returns:
            concepts_feats: [C, D] concept embeddings after projection
            avg_sim_score: [C] normalized avg similarity (0~1) per concept
            
        """
        input_ids = concepts_inputs['input_ids']         # [C, Lc]
        attention_mask = concepts_inputs['attention_mask']  # [C, Lc]
        token_type_ids = concepts_inputs['token_type_ids']  # [C, Lc]

        # The BERTEncoder here expects a packed tensor [C, 3, Lc]; double-check with your wrapper.
        concpets_feats = torch.stack([input_ids, attention_mask, token_type_ids], dim=1).to(device)  # [C, 3, Lc]
        concepts_outputs = self.text_subnet(concpets_feats).last_hidden_state[:, 0, :]  # [C, 768] use CLS token

        # Cosine similarity among concepts
        cls_norm = F.normalize(concepts_outputs, dim=1)            # [C, 768]
        sim_matrix = torch.matmul(cls_norm, cls_norm.T)            # [C, C]

        # Remove self-similarity; compute average similarity per concept
        C = sim_matrix.size(0)
        mask = ~torch.eye(C, dtype=torch.bool, device=device)      # [C, C] True on off-diagonal
        avg_sim_score = (sim_matrix * mask).sum(dim=1) / (C - 1)   # [C]
        # Min-max normalize avg_sim_score to [0,1]
        min_s, max_s = avg_sim_score.min(), avg_sim_score.max()
        avg_sim_score = (avg_sim_score - min_s) / (max_s - min_s + 1e-5)  # [C]

        # Project concepts to D for downstream attention/gating
        concepts_feats = self.concept_proj(concepts_outputs)        # [C, D]

        _concep_feats_unused = self.dropout_c(self.relu(concepts_feats))  # [C, D] (unused)

        return concepts_feats, avg_sim_score  # [C, D], [C]


    def forward(
        self,
        text_feats: torch.Tensor,
        video_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        concepts_inputs: dict,
        contrast: bool = False,
    ):
        """
        Args:
            text_feats: inputs to text_subnet (format expected by BERTEncoder wrapper)
            video_feats: [B, T, D_v] raw video features
            audio_feats: [B, T, D_a] raw audio features
            concepts_inputs: dict with 'input_ids', 'attention_mask', 'token_type_ids' (see get_concept_feats)
            contrast: whether to create masked views for contrastive training

        Returns:
            logits: [B, K] classification logits
            last_hs: [B, D] pooled (mean over T) hidden for loss heads
            concept_score: [B, C] per-sample concept scores (sum over tokens, see below)
            avg_sim_score: [C] normalized concept popularity measure (0~1)
            contrast_logits: [B or 2B, K] contrast head logits (2B if contrast=True after concat)
        """
        # Ensure float dtype for non-text modalities
        video_feats, audio_feats = video_feats.float(), audio_feats.float()

        # Text encoder
        text_outputs = self.text_subnet(text_feats)             # wrapper-defined output
        x_l = text_outputs.last_hidden_state.permute(1, 0, 2)   # [T, B, D_l] for v/a encoders later

        # Video/Audio temporal encoders
        visual = video_feats.permute(1, 0, 2)                   # [T, B, D_v]
        x_v = self.v_encoder(visual)                            # [T, B, D_v]
        acoustic = audio_feats.permute(1, 0, 2)                 # [T, B, D_a]
        x_a = self.a_encoder(acoustic)                          # [T, B, D_a]

        # Concept features
        concept_feats, avg_sim_score = self.get_concept_feats(concepts_inputs, video_feats.device)  # [C, D], [C]
        concepts_norm = F.normalize(concept_feats, dim=-1)                                          # [C, D]

        # Cross-modal fusion (returns [B, T, D])
        fusion_feats = self.fusion_net(x_l, x_v, x_a).permute(1, 0, 2)  # [B, T, D]

        # ---- Concept attention over tokens ----
        # Normalize tokens then compute cosine similarity vs concepts
        fusion_norm = F.normalize(fusion_feats, dim=2)                    # [B, T, D]
        # btd, cd -> btc 
        fusion_sim = torch.einsum('btd,cd->btc', fusion_norm, concepts_norm)  # [B, T, C]
        fusion_attn = F.softmax(fusion_sim, dim=-1)                        # [B, T, C] attention across concepts
        concept_score = fusion_sim.sum(dim=1)                               # [B, C] sum over tokens (unnormalized)

        # Optional masked view for contrastive learning
        if contrast:
            fusion_mask = self.mask_tokens(fusion_feats, fusion_attn, mask_ratio=self.mask_ratio, mode='zero')  # [B, T, D]

        # Concept-guided gating
        B = fusion_feats.size(0)
        concept_feats_b = concept_feats.unsqueeze(0).expand(B, -1, -1)     # [B, C, D]
        # Aggregate concept vectors for each token by attention
        fusion_enhanced_vec = torch.bmm(fusion_attn, concept_feats_b)      # [B, T, D]
        # Gate between original fused token and its concept aggregation
        fusion_gate_input = torch.cat([fusion_feats, fusion_enhanced_vec], dim=-1)  # [B, T, 2D]
        fusion_gate = torch.sigmoid(self.fusion_gate_fc(fusion_gate_input))         # [B, T, D]
        fusion_enhanced = fusion_gate * fusion_feats + (1.0 - fusion_gate) * fusion_enhanced_vec  # [B, T, D]

        # If contrastive, concatenate masked branch on batch dim (2B total)
        if contrast:
            fusion_enhanced = torch.cat([fusion_enhanced, fusion_mask], dim=0)  # [2B, T, D]

        # Utterance-level pooling + heads
        last_hs = self.fc(fusion_enhanced).mean(dim=1)          # [B or 2B, D], simple mean-pool over T

        contrast_logits = self.contrast_head(last_hs)           # [B or 2B, K]
        logits = self.out_layer(last_hs[:B]) if contrast else self.out_layer(last_hs)  # ensure classifier returns [B, K]

        return logits, last_hs[:B], concept_score, avg_sim_score, contrast_logits

    # Token masking
    def mask_tokens(self, x: torch.Tensor, attn_map: torch.Tensor, mask_ratio: float = 0.3, mode: str = 'zero') -> torch.Tensor:
        """
        Mask tokens with higher attention scores
        to construct harder negatives for contrastive training.
        
        """
        # Average attention over concepts -> per-token score
        attn_score = attn_map.mean(dim=-1)  # [B, T]
        B, T, D = x.shape
        masked_x = x.clone()

        for i in range(B):
            # Sort tokens by attention ascending
            sorted_indices = torch.argsort(attn_score[i])  # [T] low -> high
            num_mask = int(T * mask_ratio)

            high_attn_indices = sorted_indices[T // 2:]                      # top-50% by attention
            if len(high_attn_indices) == 0 or num_mask == 0:
                continue
            idx = torch.randperm(len(high_attn_indices))[:num_mask]
            mask_indices = high_attn_indices[idx]

            if mode == 'zero':
                masked_x[i, mask_indices] = 0
            elif mode == 'noise':
                noise = torch.randn_like(masked_x[i, mask_indices])
                masked_x[i, mask_indices] = noise
            else:
                # Fallback to zero if unknown mode
                masked_x[i, mask_indices] = 0

        return masked_x
