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
            'train_batch_size': 32,
            'eval_batch_size': 16,
            'test_batch_size': 16,
            'wait_patience': 3
        }
        return common_parameters

    def _get_hyper_parameters(self, args):
        """
            Hyper-parameters for ConceptEvo_GPT & MultimodalFusion.

            - num_train_epochs (int): training epochs
            - lr (float): learning rate
            - aligned_method (str): alignment method (for AlignSubNet)
            - hidden_dim (int): shared hidden dimension D
            - fusion_dim (int): reserved fusion dimension (optional)
            - nheads (int): attention heads
            - encoder_layers_a (int): audio encoder layers
            - encoder_layers_v (int): video encoder layers
            - vl_layers (int): cross-attn L<-V layers
            - al_layers (int): cross-attn L<-A layers
            - fusion_layers (int): fusion transformer layers
            - fusion_dropout (float): dropout after fusion
            - attn_dropout (float): attention dropout
            - relu_dropout (float): ReLU dropout
            - embed_dropout (float): embedding dropout
            - res_dropout (float): residual dropout
            - attn_mask (bool): use attention mask
            - text_dropout (float): dropout for text features
            - concept_dropout (float): dropout for concept features
            - fc_dropout_1 (float): first FC dropout
            - fc_dropout_2 (float): second FC dropout
            - grad_clip (float): gradient clip norm
            - mask_ratio (float): token mask ratio for contrastive
            - temperature (float): temperature for contrastive loss
            
        """
        hyper_parameters = {
            'num_train_epochs': 100,
            'aligned_method': 'ctc',
            'hidden_dim': 512,
            'fusion_dim': 256,
            'nheads': 8,
            'encoder_layers_a': 2,
            'encoder_layers_v': 1,
            'vl_layers': 3,
            'al_layers': 3,
            'fusion_layers': 1,
            'fusion_dropout': 0.1,
            'attn_dropout': 0.0,
            'relu_dropout': 0.1,
            'embed_dropout': 0.1, #
            'res_dropout': 0.0, #
            'attn_mask': True,
            'text_dropout': 0.4,
            'concept_dropout': 0.1,
            'fc_dropout_1': 0.1,
            'fc_dropout_2': 0.1,
            'grad_clip': 0.5, 
            'mask_ratio': 0.1,
            'temperature': 2.0, #
            'lr': 2e-5
        }
        return hyper_parameters