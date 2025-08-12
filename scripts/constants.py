DATA_PATH = 'datasets'
CFG_PATH = 'cfg'
CHECKPOINT_PATH = 'checkpoints'
EVALUATION_PATH = 'evaluation'
METADATA_PATH = 'metadata'
BRAIN_MASK = [160, 192, 160]
CFGS_RENAMING = {'age_agnostic': 'Age-agnostic',
                 'age_invariant': 'Age-invariant',
                 'age_invariant_with_age': 'Age-invariant+Age',
                 'age_aware': 'Age-aware',
                 'bmi_invariant': 'BMI-invariant',
                 'sex_invariant': 'Sex-invariant',
                 'bag': 'BAG',
                 'bag_age': 'BAG+Age',
                 'baseline': 'Random'}
PARAM_GRID = {
        'learning_rate': [0.001, 0.005, 0.01],
        'n_layers': [0, 1, 2],
        'batch_size': [8, 16, 32, 64],
        'epochs': [3, 5, 10, 15]
    }
