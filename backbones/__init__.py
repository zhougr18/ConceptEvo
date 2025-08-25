from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.ConceptEvo import ConceptEvo
from .FusionNets.MAG_BERT import MAG_BERT
from .FusionNets.MISA import MISA
from .FusionNets.MMIM import MMIM
from .FusionNets.TCL_MAP import TCL_MAP
from .FusionNets.SDIF import SDIF

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder
                }

methods_map = {
    'ConceptEvo': ConceptEvo,
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'mmim':MMIM,
    'tcl_map': TCL_MAP,
    'sdif': SDIF,
}