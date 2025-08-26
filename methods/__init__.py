from .MULT.manager import MULT
from .MAG_BERT.manager import MAG_BERT
from .MISA.manager import MISA
from .MMIM.manager import MMIM
from .TCL_MAP.manager import TCL_MAP
from .SDIF.manager import SDIF
from .ConceptEvo_GPT.manager import ConceptEvo_GPT
from .ConceptEvo.manager import ConceptEvo

method_map = {
    'mult': MULT,
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'mmim': MMIM,
    'tcl_map': TCL_MAP,
    'sdif': SDIF,
    'ConceptEvo': ConceptEvo,
    'ConceptEvo_GPT': ConceptEvo_GPT
}