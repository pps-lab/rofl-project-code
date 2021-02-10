
class AttackDatasetConfig(object):

    def __init__(self, type, train=[], test=[],
                 target_label=None,
                 remove_from_benign_dataset=False,
                 augment_times=0,
                 augment_data=False,
                 tasks=None,
                 source_label=None,
                 aux_samples=None,
                 edge_case_type=None):
        self.type = type
        self.train = train
        self.test = test
        self.source_label = source_label
        self.target_label = target_label
        self.remove_from_benign_dataset = remove_from_benign_dataset
        self.augment_times = augment_times
        self.augment_data = augment_data
        self.tasks = tasks
        self.aux_samples = aux_samples
        self.edge_case_type = edge_case_type
