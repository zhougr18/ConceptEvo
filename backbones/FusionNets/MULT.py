import torch
import torch.nn.functional as F
from ..SubNets.FeatureNets import BERTEncoder
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from torch import nn

__all__ = ['MULT']

class MULT(nn.Module):
    
    def __init__(self, args):

        super(MULT, self).__init__()
        
        self.text_subnet = BERTEncoder.from_pretrained(args.bert_base_uncased_path)

        video_feat_dim = args.video_feat_dim  # 256
        text_feat_dim = args.text_feat_dim  # 768
        audio_feat_dim = args.audio_feat_dim  # 768

        dst_feature_dims = args.dst_feature_dims  # 120

        self.orig_d_l, self.orig_d_a, self.orig_d_v = text_feat_dim, audio_feat_dim, video_feat_dim  # 768 768 256
        self.d_l = self.d_a = self.d_v = dst_feature_dims  # 120

        self.num_heads = args.nheads  # 8
        self.layers = args.n_levels # 8
        self.attn_dropout = args.attn_dropout  # 0.0
        self.attn_dropout_a = args.attn_dropout_a  # 0.2
        self.attn_dropout_v = args.attn_dropout_v  # 0.2

        self.relu_dropout = args.relu_dropout  # 0.0
        self.embed_dropout = args.embed_dropout  # 0.1
        self.res_dropout = args.res_dropout  # 0.0
        self.output_dropout = args.output_dropout  # 0.2
        self.text_dropout = args.text_dropout  # 0.4
        self.attn_mask = args.attn_mask  # True
        
        self.combined_dim = combined_dim = 2 * (self.d_l + self.d_a + self.d_v)  # 720
        output_dim = args.num_labels   # 20

        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)  # 768 120 5
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)  # 768 120 1
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)  # 256 120 1

        self.trans_l_with_a = self._get_network(self_type='la')
        self.trans_l_with_v = self._get_network(self_type='lv')

        self.trans_a_with_l = self._get_network(self_type='al')
        self.trans_a_with_v = self._get_network(self_type='av')

        self.trans_v_with_l = self._get_network(self_type='vl')
        self.trans_v_with_a = self._get_network(self_type='va')

        self.trans_l_mem = self._get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self._get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self._get_network(self_type='v_mem', layers=3)

        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
    def _get_network(self, self_type='l', layers=-1):

        if self_type in ['l', 'vl', 'al']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a  # 120 0.2
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,  # 8
                                  layers=max(self.layers, layers), # 8
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text_feats, video_feats, audio_feats):
        video_feats, audio_feats = video_feats.float(), audio_feats.float()        
        
        text_outputs = self.text_subnet(text_feats)
        text = text_outputs.last_hidden_state    # [16,30,768]

        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)  # [16,768,30]  p=0.4
        x_a = audio_feats.transpose(1, 2)  # [16,768,480]<-[16,480,768]  转置
        x_v = video_feats.transpose(1, 2)  # [16,256,230]<-[16,230,256]

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)  # [16,120,26] text
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)  # [16,120,480] audio
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)  # [16,120,230] video

        proj_x_l = proj_x_l.permute(2, 0, 1)  # [26,16,120]
        proj_x_a = proj_x_a.permute(2, 0, 1)  # [480,16,120]
        proj_x_v = proj_x_v.permute(2, 0, 1)  # [230,16,120]

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # [26,16,120]
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # [26,16,120]  # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim = 2)  # [26,16,240]
        h_ls = self.trans_l_mem(h_ls)  # [26,16,240]
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]   # [16,240] Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)  # [480,16,120]
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)  # [480,16,120]
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)  # [480,16,240]
        h_as = self.trans_a_mem(h_as)  # [480,16,240]
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]  # [16,240] 取最后一个时刻

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)  # [230,16,120]
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)  # [230,16,120]
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)  # [230,16,240]
        h_vs = self.trans_v_mem(h_vs)  # [230,16,240]
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]  # [16,240]

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)  # [16,720]
        
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))  # [16,720]
        last_hs_proj += last_hs  # [16,720]
        
        logits = self.out_layer(last_hs_proj)  # [16,20]

        return logits, last_hs