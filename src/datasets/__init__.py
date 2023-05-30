from . import data_module, dataset_coco, dataset_generic

ALL_DATASETS = {
    'coco': dataset_coco.ClipCocoDataset,
    'video': dataset_generic.GenericDataset,
    'medical': dataset_generic.GenericDataset,
    'amino_acid': dataset_generic.GenericDataset,
    'audio': dataset_generic.GenericDataset
}