import numpy as np
from onmt_utils.evaluator import get_base_names_from_folder
from abc import abstractmethod
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
DrawingOptions.includeAtomNumbers=True


def get_names(base_names):
    return base_names


def assign_left(data_sorted, cdf, x_grid):
    indexes = []
    new_cdf = np.zeros(len(x_grid))

    for k, elem in enumerate(x_grid):
        i = 0
        while i < len(data_sorted) and elem >= data_sorted[i]:
            i += 1
        indexes.append(i-1)
    for k, idx in enumerate(indexes):
        if idx == -1:
            new_cdf[k] = 0.0
        else:
            new_cdf[k] = cdf[idx]
    return new_cdf


class EvalModel():
    def __init__(self, datapath, dataset_name, split='test', n_best=1, class_sep=' '):

        self.datapath = datapath                         # path to the dataset files
        self.dataset_name = dataset_name                 # name of the dataset
        self.n_best = n_best                             # number of predictions performed
        self.name_folder_dict = {}                       # dict to store names and correspondent folder 
        self.split = split                               # dataset split
        self.name_basename_dict = {}                     # dict to store names and correspondent basenames
        self.split_evaluator = None

        # Add the precursors of the target set
        with open(datapath + f'/precursors-{split}.txt', 'r') as f:
            self.precursors_tok = [line.strip() for line in f.readlines()]
            self.precursors = [elem.replace(" ", "") for elem in self.precursors_tok]

        # Add the product of the target set
        with open(datapath + f'/product-{split}.txt', 'r') as f:
            self.product_tok = [line.strip() for line in f.readlines()]
            self.product = [elem.replace(" ", "") for elem in self.product_tok]

        # Add the class information for the predictions
        with open(datapath + f'/class-multi-{self.split}.txt', 'r') as f:
            self.classes_tok = [line.strip() for line in f.readlines()]
            self.classes = [elem.split(class_sep) for elem in self.classes_tok]
 
        self.__create_evaluators()

    @abstractmethod
    def __create_evaluators(self):
        pass
    
    def add_experiment(self, results_path, func=get_names):
     
        base_names = get_base_names_from_folder(results_path)
        names = func(base_names)
        
        for i, exp_name in enumerate(names):
            self.split_evaluator.append_experiment(results_path, base_names[i], exp_name, n_best=self.n_best)
            
            for n in range(1, self.n_best +1):
                self.split_evaluator.df[f'{exp_name}_top_{n}_valid'] = self.split_evaluator.df[f'{exp_name}_top_{n}'] != self.split_evaluator.invalid_smiles_replacement
             
            self.name_folder_dict[exp_name] = results_path
            self.name_basename_dict[exp_name] = base_names[i]

        return names
        
    def print_experiments(self):
        print(f'Experiments list for split {self.split}: ')
        for key in sorted(self.name_basename_dict):
            print(key)

    def get_exp_top_n_accuracy(self, exp_name, topn=1, hashing=None):
        acc = 0.0
        if exp_name in self.name_basename_dict:
            if hashing != None:
                acc = self.split_evaluator.get_top_n_accuracy_with_hashing(exp_name, n=topn, hashing=hashing)
            else:
                acc = self.split_evaluator.get_top_n_accuracy(exp_name, n=topn)
        else:
            print('Sorry, this experiment is not present')
        return acc

    def get_allexp_top_n_accuracy(self, topn=1, hashing=None):
        acc = []
        labels = []
        for exp_name in self.name_basename_dict:
            labels.append(exp_name)
            if hashing != None:
                acc.append(self.get_exp_top_n_accuracy(exp_name, topn=topn, hashing=hashing))
            else:
                acc.append(self.get_exp_top_n_accuracy(exp_name, topn=topn))
        return acc, labels
