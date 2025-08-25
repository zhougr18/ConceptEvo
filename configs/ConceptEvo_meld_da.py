class Param():
    
    def __init__(self, args):
        
        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):
        """
            padding_mode (str): The mode for sequence padding ('zero' or 'normal').
            padding_loc (str): The location for sequence padding ('start' or 'end'). 
            eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').  
            need_aligned: (bool): Whether to perform data alignment between different modalities.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patience (int): Patient steps for Early Stop.
        """
        common_parameters = {
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': False,
            'eval_monitor': 'f1',
            'train_batch_size': [16],
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 3
        }
        return common_parameters

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            dst_feature_dims (int): The destination dimensions (assume d(l) = d(v) = d(t)).
            nheads (int): The number of heads for the transformer network.
            n_levels (int): The number of layers in the network.
            attn_dropout (float): The attention dropout.
            attn_dropout_v (float): The attention dropout for the video modality.
            attn_dropout_a (float): The attention dropout for the audio modality.
            relu_dropout (float): The relu dropout.
            embed_dropout (float): The embedding dropout.
            res_dropout (float): The residual block dropout.
            output_dropout (float): The output layer dropout.
            text_dropout (float): The dropout for text features.
            grad_clip (float): The gradient clip value.
            attn_mask (bool): Whether to use attention mask for Transformer. 
            conv1d_kernel_size_l (int): The kernel size for temporal convolutional layers (text modality).  
            conv1d_kernel_size_v (int):  The kernel size for temporal convolutional layers (video modality).  
            conv1d_kernel_size_a (int):  The kernel size for temporal convolutional layers (audio modality).  
            lr (float): The learning rate of backbone.
        """
        hyper_parameters = {
            'num_train_epochs': 100,
            'lr': [1e-5],
            'aligned_method': 'ctc',
            'hidden_dim': [1024],
            'fusion_dim': [1024],
            'nheads': 8,
            'encoder_layers_a': [1],
            'encoder_layers_v': [3],
            'vl_layers': [4],
            'al_layers': [2],
            'fusion_layers': [2],
            'fusion_dropout': [0.1],
            'attn_dropout': [0.0],
            'relu_dropout': [0.0],
            'embed_dropout': [0.1],
            'res_dropout': [0.0],
            'attn_mask': True,
            'text_dropout': 0.4,
            'concept_dropout': [0.1],
            'fc_dropout_1': [0.1],
            'fc_dropout_2': [0.1],
            'grad_clip': [0.5], 
            'mask_ratio': [0.35],
            'temperature': [1.15],
        }
        return hyper_parameters