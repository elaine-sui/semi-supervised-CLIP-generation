from enum import Enum

class MappingType(str, Enum):
    MLP = 'mlp'
    Transformer = 'transformer'
    Linear = 'linear'

class Modality(str, Enum):
    Vision = 'vision'
    Language = 'language'
    Both = 'both'

class DatasetType(str, Enum):
    Video = 'video'
    Medical = 'medical'
    Amino_Acid = 'amino_acid'
    Audio = 'audio'