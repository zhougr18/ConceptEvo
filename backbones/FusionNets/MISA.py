import torch
from torch import nn
from torch.autograd import Function
from ..SubNets.FeatureNets import BERTEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ['MISA']

class ReverseLayerF(Function):
    """
    Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
    """
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

class MISA(nn.Module):
    
    def __init__(self, args):

        super(MISA, self).__init__()

        self.text_subnet = BERTEncoder.from_pretrained(args.bert_base_uncased_path)
        self.visual_size = args.video_feat_dim   # 256
        self.acoustic_size = args.audio_feat_dim  # 768
        self.text_size = args.text_feat_dim  # 768

        self.dropout_rate = args.dropout_rate  # 0.1
        self.output_dim = args.num_labels  # 20
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]  # [768,256,768]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]  # [768,256,768]
        self.args = args
        
        rnn = nn.LSTM if args.rnncell == "lstm" else nn.GRU

        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)  # LSTM(256, 256, bidirectional=True)
        self.vrnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0], out_features=args.hidden_size))  # 768 256
        self.project_t.add_module('project_t_activation', self.activation)  # ReLU()
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(args.hidden_size))  # 256
        
        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1] * 4, out_features=args.hidden_size))  # 1024 256
        self.project_v.add_module('project_v_activation', self.activation)  # ReLU()
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(args.hidden_size))  # 256

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2] * 4, out_features=args.hidden_size))  # 3072 256
        self.project_a.add_module('project_a_activation', self.activation)  # ReLU()
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(args.hidden_size))  # 256

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))  # 256 256
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())  # Sigmoid
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))  # 256 256
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())  # Sigmoid

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))  # 256 256
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())  # Sigmoid

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))  # 256 256
        self.shared.add_module('shared_activation_1', nn.Sigmoid())  # Sigmoid

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))  # 256 256
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))  # 256 256
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))  # 256 256

        if not args.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size)) # 256 256
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation) # ReLU
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(args.dropout_rate)) # 0.1
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=args.hidden_size, out_features=len(hidden_sizes))) # 256 3

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=args.hidden_size, out_features = 4))  # 256 4

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=args.hidden_size * 6, out_features=args.hidden_size * 3))  # 1536 768
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))  # 0.1
        self.fusion.add_module('fusion_layer_1_activation', self.activation)  # ReLU()
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=args.hidden_size * 3, out_features= self.output_dim))  # 768 20

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
   
    def _reconstruct(self):
    
        self.utt_t = (self.utt_private_t + self.utt_shared_t)  # [16,256] 
        self.utt_v = (self.utt_private_v + self.utt_shared_v)  # [16,256]
        self.utt_a = (self.utt_private_a + self.utt_shared_a)  # [16,256]

        self.utt_t_recon = self.recon_t(self.utt_t)  # [16,256] 
        self.utt_v_recon = self.recon_v(self.utt_v)  # [16,256] 
        self.utt_a_recon = self.recon_a(self.utt_a)  # [16,256] 

    def _shared_private(self, utterance_t, utterance_v, utterance_a):   # [16,768]  [16,1024]  [16,3072]
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)  # [16,256]
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)  # [16,256] 
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)  # [16,256]

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)  # [16,256]
        self.utt_private_v = self.private_v(utterance_v)  # [16,256]
        self.utt_private_a = self.private_a(utterance_a)  # [16,256]

        self.utt_shared_t = self.shared(utterance_t)  # [16,256]
        self.utt_shared_v = self.shared(utterance_v)  # [16,256]
        self.utt_shared_a = self.shared(utterance_a)  # [16,256]
    
    def _extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):

        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        
        if self.args.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)  # [20,16,512]
        padded_h1 = padded_h1.permute(1, 0, 2)  # [16,20,512]
        normed_h1 = layer_norm(padded_h1)  # [16,20,512]
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)

        if self.args.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)  # [2,16,256]
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def forward(self, text_feats, video_feats, audio_feats):
        video_feats, audio_feats = video_feats.float(), audio_feats.float()
        batch_size = text_feats.size(0)
        bert_sent, bert_sent_mask, bert_sent_type = text_feats[:,0], text_feats[:,1], text_feats[:,2]  # [16,30]  [16,30]  [16,30]
        bert_output = self.text_subnet(text_feats)
        bert_output = bert_output.last_hidden_state   # [batch_size, seq_len, 768]  # [16,30,768]

        # Use the mean value of bert of the front real sentence length as the final representation of text.
        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)  # [16,30,768]
        mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  # [16,1]
        text_feats = torch.sum(masked_output, dim=1, keepdim=False) / mask_len  # [16,768]  [16,768]/[16,1]

        lengths = mask_len.squeeze().int().detach().cpu().view(-1)  # 16

        # extract features from visual modality
        final_h1v, final_h2v = self._extract_features(video_feats, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        video_feats = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)  # [16,1024]

        # extract features from acoustic modality
        final_h1a, final_h2a = self._extract_features(audio_feats, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        audio_feats = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)  # [16,3072]

        self._shared_private(text_feats, video_feats, audio_feats)
        
        if not self.args.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.args.reverse_grad_weight)  # [16,256]
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.args.reverse_grad_weight)  # [16,256]
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.args.reverse_grad_weight)  # [16,256]

            self.domain_label_t = self.discriminator(reversed_shared_code_t)  # [16,3]
            self.domain_label_v = self.discriminator(reversed_shared_code_v)  # [16,3]
            self.domain_label_a = self.discriminator(reversed_shared_code_a)  # [16,3]
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)  # [16,4]
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)  # [16,4]
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)  # [16,4]
        self.shared_or_private_s = self.sp_discriminator((self.utt_shared_t + self.utt_shared_v + self.utt_shared_a) / 3.0)  # [16,4]

        self._reconstruct()

        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0) # [6,16,256]
        h = self.transformer_encoder(h)  # [6,16,256]
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)  # [16,1536]
        logits = self.fusion(h)  # [16,20]

        return logits