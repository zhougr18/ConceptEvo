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
            'eval_monitor': ['f1'],
            'train_batch_size': 16,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8,
            'num_train_epochs': 100,
        }
        return common_parameters
    
    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            beta_shift (float): The coefficient for nonverbal displacement to create the multimodal vector.
            dropout_prob (float): The embedding dropout probability.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr (float): The learning rate of backbone.
            aligned_method (str): The method for aligning different modalities. ('ctc', 'conv1d', 'avg_pool')
            weight_decay (float): The coefficient for L2 regularization. 
        """
        hyper_parameters = {
            'gamma': [1],
            'theta': [1],
            'add_va': False,
            'cpc_activation': 'Tanh',
            'mmilb_mid_activation': 'ReLU',
            'mmilb_last_activation': 'Tanh',
            'optim': 'Adam',
            'contrast': True,
            'bidirectional': True,
            'grad_clip': [0.9],
            'lr_main': [2e-5],
            'weight_decay_main': [0.0001],
            'lr_bert': [2e-5],
            'weight_decay_bert': [0.001],
            'lr_mmilb': [0.001],
            'weight_decay_mmilb': [0.0004],
            'alpha': [0.4],
            'beta': [0.1],
            'dropout_a': [0.2],   #0.1
            'dropout_v': [0.2],  #0.1
            'dropout_prj': [0.2],
            'n_layer': [2],
            'cpc_layers': [4],
            'd_vh': [8],
            'd_ah': [4],
            'd_vout': [4],
            'd_aout': [4],
            'd_prjh': [512],
            'scale': [20],
            'label_len': 3,
        }
        return hyper_parameters