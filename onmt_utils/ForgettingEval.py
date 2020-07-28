import math
from onmt_utils.ForwardEvalModel import *
from onmt_utils.RetroEvalModel import *


def get_names(base_names):
    return ['_'.join(b.split('_')[-1]) for b in base_names]


class ForgettingForward(ForwardEvalModel):
    def __init__(self, datapath, dataset_name, split='train', n_best=1, class_sep=' '):
        ForwardEvalModel.__init__(self, datapath, dataset_name, split=split, n_best=n_best, class_sep=class_sep)

    def calculate_forgotten_events(self):
        self.split_evaluator.df[f'forgotten_events'] = 0
        self.split_evaluator.df[f'old_state'] = False  # indicates if the network learns that example or not
        self.split_evaluator.df[f'first_learnt'] = math.inf
        cols_to_int = [int(exp_name) for exp_name in self.name_basename_dict]
        cols = [str(exp_name) for exp_name in sorted(cols_to_int)]

        df = self.split_evaluator.df

        # define when the examples are first learnt
        for i, step in enumerate(cols):
            df.loc[(df[f'first_learnt'] == math.inf) & (df[f'{step}_top_1_correct'] == True), f'first_learnt'] = int(i)

        for step in cols:
            df.loc[df['old_state'] & (df[f'{step}_top_1_correct'] == False), f'forgotten_events'] += 1
            df['old_state'] = df[f'{step}_top_1_correct']
        
        return cols

class ForgettingRetro(RetroEvalModel):
    def __init__(self, datapath, dataset_name, split='train', n_best=1, class_sep=' '):
        RetroEvalModel.__init__(self, datapath, dataset_name, split=split, n_best=n_best, class_sep=class_sep)

    def calculate_forgotten_events(self):
        self.split_evaluator.df[f'forgotten_events'] = 0
        self.split_evaluator.df[f'old_state'] = False  # indicates if the network learns that example or not
        self.split_evaluator.df[f'first_learnt'] = math.inf
        cols_to_int = [int(exp_name) for exp_name in self.name_basename_dict]
        cols = [str(exp_name) for exp_name in sorted(cols_to_int)]

        df = self.split_evaluator.df

        # select the prediction with the highest likelihood value
        for step in cols:
            df[f'{step}_topn'] = 1
            prob_cols = [f'fwd_{step}_top_{k}_prob' for k in range(1, self.n_best + 1)]
            selection = df[prob_cols].idxmax(axis=1)
            df[f'{step}_topn'] = selection.map(lambda x: x.split('_')[-2])
            df[f'{step}_topn_correct'] = False
            for k in range(1, self.n_best + 1):
                df.loc[df[f'{step}_topn'] == str(k), f'{step}_topn_correct'] = \
                    df.loc[df[f'{step}_topn'] == str(k), f'fwd_{step}_top_{k}_correct']

        # define when the examples are first learnt
        for i, step in enumerate(cols):
            df.loc[(df[f'first_learnt'] == math.inf) & (df[f'{step}_topn_correct'] == True), f'first_learnt'] = int(i)

        for step in cols:
            df.loc[df['old_state'] & (df[f'{step}_topn_correct'] == False), f'forgotten_events'] += 1
            df['old_state'] = df[f'{step}_topn_correct']

        return cols
