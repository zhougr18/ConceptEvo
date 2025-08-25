import torch
import torch.nn.functional as F
from ..SubNets.FeatureNets import BERTEncoder
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from ..SubNets.AlignNets import AlignSubNet
from torch import nn


def get_transformer_encoder(args, embed_dim, layers):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=args.nheads,
                                  layers=layers,
                                  attn_dropout=args.attn_dropout,
                                  relu_dropout=args.relu_dropout,
                                  res_dropout=args.res_dropout,
                                  embed_dropout=args.embed_dropout,
                                  attn_mask=args.attn_mask)

class MultimodalFusion(nn.Module):
    def __init__(self, args):
        super(MultimodalFusion, self).__init__()
        self.v_proj = nn.Linear(args.video_feat_dim, args.hidden_dim)
        self.a_proj = nn.Linear(args.audio_feat_dim, args.hidden_dim)
        self.l_proj = nn.Linear(args.text_feat_dim, args.hidden_dim)
        self.align_net = AlignSubNet(args, args.aligned_method)
        self.cross_attn_vl = get_transformer_encoder(args, args.hidden_dim, args.vl_layers)
        self.cross_attn_al = get_transformer_encoder(args, args.hidden_dim, args.al_layers)

        # 融合模块
        self.fusion = nn.Linear(3 * args.hidden_dim, args.hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(args.fusion_dropout)

        # Optional: final transformer encoder
        self.final_encoder = get_transformer_encoder(args, args.hidden_dim, args.fusion_layers)
        

    def forward(self, x_l, x_v, x_a):

        x_l, x_a, x_v  = self.align_net(x_l.permute(1, 0, 2), x_a.permute(1, 0, 2), x_v.permute(1, 0, 2))

        x_l, x_a, x_v = x_l.permute(1, 0, 2), x_a.permute(1, 0, 2), x_v.permute(1, 0, 2)

        x_l = self.l_proj(x_l)  # [B, T, D]
        x_v = self.v_proj(x_v)  # [B, T, D]
        x_a = self.a_proj(x_a)  # [B, T, D]

        x_vl = self.cross_attn_vl(x_l, x_v, x_v) 
        x_al = self.cross_attn_al(x_l, x_a, x_a)

        fusion_feats = torch.cat([x_l, x_vl, x_al], dim=-1)  # [B, T, 3D]
        fusion_feats = self.fusion(fusion_feats)  # [B, T, D]
        fusion_feats = self.activation(fusion_feats)
        fusion_feats = self.dropout(fusion_feats)

        # Step 4 (optional): further transformer encoding
        fusion_feats = self.final_encoder(fusion_feats)  # [B, T, D]

        return fusion_feats



class ConceptEvo(nn.Module):
    
    def __init__(self, args):

        super(MULT, self).__init__()
        
        self.text_subnet = BERTEncoder.from_pretrained(args.bert_base_uncased_path)

        self.video_feat_dim = args.video_feat_dim  # 256
        self.text_feat_dim = args.text_feat_dim  # 768
        self.audio_feat_dim = args.audio_feat_dim  # 768
        self.hidden_dim = args.hidden_dim  # 120

        self.mask_ratio = args.mask_ratio  # 0.2

        self.v_encoder = get_transformer_encoder(args, args.video_feat_dim, args.encoder_layers_v)
        self.a_encoder = get_transformer_encoder(args, self.audio_feat_dim, args.encoder_layers_a)


        self.concept_proj = nn.Linear(args.text_feat_dim, args.hidden_dim)  # 768
        self.relu = nn.ReLU()
        self.dropout_c = nn.Dropout(p=args.concept_dropout)  # 0.1

        self.fusion_net = MultimodalFusion(args)

        self.fusion_gate_fc = nn.Linear(args.hidden_dim * 2, args.hidden_dim)


        self.fc = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),  # 120 -> 240
            nn.ReLU(),
            nn.Dropout(p=args.fc_dropout_1),  # 0.1
            nn.Linear(args.hidden_dim // 2, args.hidden_dim),  # 240 -> 360
            nn.ReLU(),
            nn.Dropout(p=args.fc_dropout_2)  # 0.1
        )

        self.out_layer = nn.Linear(args.hidden_dim, args.num_labels)
    
        

    def get_concept_feats(self, concepts_inputs, device):
        """
        Get the concept features from the text subnet.
        This method is used to extract the concept features for the concepts inputs.
        """
        input_ids = concepts_inputs['input_ids']  # [6,4]
        attention_mask = concepts_inputs['attention_mask']  # [6,4]
        token_type_ids = concepts_inputs['token_type_ids']  # [6,4]  
        concpets_feats = torch.stack([input_ids, attention_mask, token_type_ids], dim=1).to(device)  # [6,3,4]
        # concepts_outputs = self.text_subnet(concpets_feats).last_hidden_state.mean(dim=1)  # [6,768] 取平均值
        concepts_outputs = self.text_subnet(concpets_feats).last_hidden_state[:,0,:]  # [6,768] 取平均值

        cls_norm = F.normalize(concepts_outputs, dim=1)  # [B, D]
        sim_matrix = torch.matmul(cls_norm, cls_norm.T)  # [B, B]，对称矩阵，表示任意 concept 两两之间相似度

        # Step 4: 去除自身（对角线）影响，计算每个 concept 的平均相似度
        B = sim_matrix.shape[0]
        mask = ~torch.eye(B, dtype=torch.bool, device=device)  # 取出非对角线部分
        avg_sim_score = (sim_matrix * mask).sum(dim=1) / (B - 1)
        min_s, max_s = avg_sim_score.min(), avg_sim_score.max()
        avg_sim_score = (avg_sim_score - min_s) / (max_s - min_s + 1e-5)
        

        concepts_feats = self.concept_proj(concepts_outputs)
        concepts_feats = self.dropout_c(self.relu(concepts_feats))  # [6,120]
        return concepts_feats, avg_sim_score

    def forward(self, text_feats, video_feats, audio_feats, concepts_inputs, contrast=False):
        video_feats, audio_feats = video_feats.float(), audio_feats.float()   
        
        text_outputs = self.text_subnet(text_feats)
        x_l = text_outputs.last_hidden_state.permute(1, 0, 2)
        visual = video_feats.permute(1, 0, 2)
        x_v = self.v_encoder(visual)  
        acoustic = audio_feats.permute(1, 0, 2)
        x_a = self.a_encoder(acoustic)

        concept_feats, avg_sim_score = self.get_concept_feats(concepts_inputs, video_feats.device)  # [6,120]
        concepts_norm = F.normalize(concept_feats, dim=-1)  # [6, 120]

        fusion_feats = self.fusion_net(x_l, x_v, x_a).permute(1, 0, 2)  # [B, T, D]

        # Step 2: 计算余弦相似度
        # 结果 shape: [batch, seq_len, concepts]
        fusion_norm = F.normalize(fusion_feats, dim=2)    # [16, 120, 26]
        
        fusion_sim = torch.einsum('btd,cd->btc', fusion_norm, concepts_norm)        # [16, 26, 6]
        fusion_attn = F.softmax(fusion_sim, dim=-1)  # [B, T, C]
        concept_score = fusion_sim.sum(dim=1)

        B = fusion_feats.size(0)  # batch size
        concept_feats = concept_feats.unsqueeze(0).expand(B, -1, -1) 
        fusion_enhanced_vec = torch.bmm(fusion_attn, concept_feats)

        fusion_gate_input = torch.cat([fusion_feats, fusion_enhanced_vec], dim=-1)  # [B, T, 2D]
        fusion_gate = torch.sigmoid(self.fusion_gate_fc(fusion_gate_input))  # [B, T, D]
        fusion_enhanced = fusion_gate * fusion_feats + (1 - fusion_gate) * fusion_enhanced_vec

        last_hs = self.fc(fusion_enhanced).mean(dim=1)  # [B, T, D]
        

        logits = self.out_layer(last_hs)

        return logits, last_hs, concept_score, avg_sim_score
