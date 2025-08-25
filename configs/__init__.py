video_feats_map = {
    'video_feats.pkl': 'resnet-50', # mintrec
    'swin_roi.pkl': 'swin-roi',     # mintrec2
    'swin_feats.pkl': 'swin-full',  # mintrec(tcl-map) iemocap-da meld-da
}

audio_feats_map = {
    'wavlm_feats.pkl': 'wavlm',     # mintrec mintrec2 iemocap-da meld-da
    'audio_feats.pkl': 'wav2vec2',  # mintrec
}

feat_dims = {
    'text': {
        'bert-base-uncased': 768,
        'bert-large-uncased': 1024
    },
    'video': {
        'resnet-50': 256,
        'swin-roi': 256,
        'swin-full': 1024,    
    },
    'audio': {
        'wavlm': 768,
        'wav2vec2': 768
    }
}