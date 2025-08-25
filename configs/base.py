import importlib
from easydict import EasyDict
from .__init__ import video_feats_map, audio_feats_map, feat_dims

class ParamManager:
    
    def __init__(self, args):

        args = self._add_param(args)
        
        common_param, hyper_param = self._get_config_param(args)

        self.args = EasyDict(
                                dict(
                                        vars(args),
                                        **common_param,
                                        **hyper_param
                                     )
                            )

    def _get_config_param(self, args):
        
        if args.config_file_name.endswith('.py'):
            module_name = '.' + args.config_file_name[:-3]
        else:
            module_name = '.' + args.config_file_name

        config = importlib.import_module(module_name, 'configs')

        config_param = config.Param
        method_args = config_param(args)

        return method_args.common_param, method_args.hyper_param
    
    def _add_param(self, args):
        args.text_feat_dim = feat_dims['text'][args.text_backbone]
        args.video_feat_dim = feat_dims['video'][video_feats_map[args.video_feats_path]]
        args.audio_feat_dim = feat_dims['audio'][audio_feats_map[args.audio_feats_path]]

        return args