"""
Generates the rxns files necessary for training and testing.
The data needs to be in the following format (columns):
    ['original_rxn', 'rxn_class', 'source', 'rxn', 'split']
"""

import pandas as pd
import random
import numpy as np
import os
from rdkit import RDLogger
from onmt_utils.computation_utils.smiles_utils import convert_to_tokenized_canonical_smiles

RDLogger.DisableLog('rdApp.*')
script_dir = os.path.dirname(__file__)


def main():
    df = pd.read_csv(script_dir + filename)

    df['precursors'] = df['can_reaction'].apply(lambda x: x.split('>>')[0])
    df['product'] = df['can_reaction'].apply(lambda x: x.split('>>')[1])
    print('Tokenizing the reactions ...')
    df['tok_precursors'] = df['precursors'].apply(lambda x: convert_to_tokenized_canonical_smiles(x))
    df['tok_product'] = df['product'].apply(lambda x: convert_to_tokenized_canonical_smiles(x))
    df['superclass'] = df['rxn_Class'].apply(lambda x: str(x))

    print(df.loc[0].values)
    print(df.head(), df.columns)

    # Generate train/test split
    split_ratio = 0.1

    random.seed(42)
    np.random.seed(42)
    number_of_unique_products = len(df['tok_product'].unique())
    sample = np.split(pd.Series(df['tok_product'].unique()).sample(frac=1, random_state=42),
                      [int((1.0 - split_ratio) * number_of_unique_products)])

    train_df = df.loc[df['tok_product'].isin(sample[0])]
    test_df = df.loc[df['tok_product'].isin(sample[1])]

    print(len(df))
    print('# Train examples: ', len(train_df))
    print('# Test examples: ', len(test_df))

    print('Generating files ...')
    with open(dest_path + 'precursors-test.txt', 'w') as f:
        f.write('\n'.join(test_df['tok_precursors'].values))
    with open(dest_path + 'product-test.txt', 'w') as f:
        f.write('\n'.join(test_df['tok_product'].values))
    with open(dest_path + 'class-multi-test.txt', 'w') as f:
        f.write('\n'.join(test_df['superclass'].values))
    with open(dest_path + 'class-single-test.txt', 'w') as f:
        f.write('\n'.join(test_df['superclass'].values))

    with open(dest_path + 'precursors-train.txt', 'w') as f:
        f.write('\n'.join(train_df['tok_precursors'].values))
    with open(dest_path + 'product-train.txt', 'w') as f:
        f.write('\n'.join(train_df['tok_product'].values))
    with open(dest_path + 'class-multi-train.txt', 'w') as f:
        f.write('\n'.join(train_df['superclass'].values))
    with open(dest_path + 'class-single-train.txt', 'w') as f:
        f.write('\n'.join(train_df['superclass'].values))


if __name__ == '__main__':
    # THIS PART is to be changed if the dataset changes
    filename = "/../noise_data/schneider50k/schneider_can_rxn_50k.csv"
    dest_path = script_dir + "/../noise_data/schneider50k/"

    main()
