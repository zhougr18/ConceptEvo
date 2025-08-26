from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.ConceptEvo import ConceptEvo
from .FusionNets.ConceptEvo_GPT import ConceptEvo_GPT
from .FusionNets.MAG_BERT import MAG_BERT
from .FusionNets.MISA import MISA
from .FusionNets.MULT import MULT
from .FusionNets.MMIM import MMIM
from .FusionNets.TCL_MAP import TCL_MAP
from .FusionNets.SDIF import SDIF

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder
                }

methods_map = {
    'ConceptEvo': ConceptEvo,
    'ConceptEvo_GPT': ConceptEvo_GPT,
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'MULT': MULT,
    'mmim':MMIM,
    'tcl_map': TCL_MAP,
    'sdif': SDIF,
}