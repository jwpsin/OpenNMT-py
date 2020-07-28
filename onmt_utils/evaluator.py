# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import logging
from rdkit.Chem.rdMolHash import MolHash, HashFunction
from rdkit import Chem


def get_base_names_from_folder(folder):
    base_names = []
    for f in os.listdir(folder):
        file_name = f.split('/')[-1]
        if file_name.startswith('can_'):
            base_names.append(f[4:])
    return base_names

class ONMTMultiEvaluator:
    def __init__(self,
                 targets,
                 sources=None,
                 logger=None,
                 invalid_smiles_replacement='C.C'
                 ):

        self.df = pd.DataFrame(targets)
        self.df.columns = ['target']

        if sources is not None:
            self.df['source'] = sources

        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.names = []
        self.names_to_show = []

        self.invalid_smiles_replacement = invalid_smiles_replacement

        self.total = len(self.df)

    def append_experiment(self,
                          result_folder,
                          prediction_base,
                          name=None,
                          gold_score=True,
                          log_likelihood=True,
                          n_best=1
                          ):
        experiment_name = name if name is not None else prediction_base

        canonical_prediction_file = os.path.join(result_folder, 'can_' + prediction_base)

        predictions = [[] for i in range(n_best)]
        with open(canonical_prediction_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                predictions[i % n_best].append(line.strip().replace(" ", ""))

        for n in range(1, n_best + 1):
            self.df[f'{experiment_name}_top_{n}'] = predictions[n - 1]
            self.df[f'{experiment_name}_top_{n}_correct'] = self.df['target'] == self.df[f'{experiment_name}_top_{n}']

        if gold_score:
            gold_score_file = os.path.join(result_folder, prediction_base + '_gold_score')
            self.logger.info(f"Loading gold scores from {gold_score_file}")
            with open(gold_score_file, 'r') as f:
                gold_nll = [-float(line.strip()) for line in f.readlines()]

            self.df[f'{experiment_name}_nll'] = gold_nll
            self.df[f'{experiment_name}_prob'] = [np.exp(-nll) for nll in gold_nll]

        if log_likelihood:
            log_probs_file = os.path.join(result_folder, prediction_base + '_log_probs')
            self.logger.info(f"Loading log probabilities from {log_probs_file}")

            nll_predictions = [[] for i in range(n_best)]
            probas_predictions = [[] for i in range(n_best)]

            with open(log_probs_file, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    nll_predictions[i % n_best].append(-float(line.strip()))
                    probas_predictions[i % n_best].append(np.exp(float(line.strip())))

            for n in range(1, n_best + 1):
                self.df[f'{experiment_name}_top{n}_nll'] = nll_predictions[n - 1]
                self.df[f'{experiment_name}_top{n}_prob'] = probas_predictions[n - 1]

        self.names.append(experiment_name)
        self.names_to_show.append(experiment_name)

    def get_top_n_accuracy(self, name, n=1):
        correct = 0
        df_filter = [True for i in range(len(self.df))]
        for i in range(1, n + 1):
            correct += self.df[df_filter][f'{name}_top_{i}_correct'].sum()
            df_filter = df_filter & (self.df[f'{name}_top_{i}_correct'] == False)

        return correct / len(self.df) * 100

    def get_top_n_accuracy_with_hashing(self, name, n=1, hashing=HashFunction.HetAtomTautomer):
        """
        Applies the specified hashing to account for protomers, tautomers, REDOX
        """
        correct = 0
        df_filter = [True for i in range(len(self.df))]
        self.hashing = hashing

        """
        Hash the target
        """
        target_no_frag = [elem.replace("~", ".").split('.') for elem in self.df.target]
        hashed_target = [[MolHash(Chem.MolFromSmiles(elem[i]), hashing) for i in range(len(elem))] for elem in
                         target_no_frag]
        self.df['hashed_target'] = [".".join(elem) for elem in hashed_target]

        """
        Hash the predictions
        """
        for i in range(1, n + 1):
            prediction_no_frag = [elem.replace("~", ".").split('.') for elem in self.df[f'{name}_top_{i}']]
            pred = [[MolHash(Chem.MolFromSmiles(elem[i]), hashing) for i in range(len(elem))] for elem in
                    prediction_no_frag]
            self.df[f'hashed_{name}_top_{i}'] = [".".join(elem) for elem in pred]
            self.df[f'hashed_{name}_top_{i}_correct'] = self.df[f'hashed_{name}_top_{i}'] == self.df['hashed_target']
            correct += self.df[df_filter][f'hashed_{name}_top_{i}_correct'].sum()
            df_filter = df_filter & (self.df[f'hashed_{name}_top_{i}_correct'] == False)

        return correct / len(self.df) * 100
