from enum import Enum

class MappingType(str, Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class Modality(str, Enum):
    Vision = 'vision'
    Language = 'language'
    Both = 'both'