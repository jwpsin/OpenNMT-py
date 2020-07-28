import argparse
from rdkit import Chem
import os

def canonicalize_basic(smi: str):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), True)


def canonicalize(smi: str) -> str:
    if smi[0] in ['.', '~'] or smi[-1] in ['.', '~']:
        Chem.MolToSmiles(Chem.MolFromSmiles('(C'), True)
    elif '~' in smi:
        res = '.'.join(sorted([canonicalize_basic(_.replace('~','.')).replace('.','~')
                                  for _ in smi.split('.')]))
        return res
    else:
        return canonicalize_basic(smi)

# tokenize function

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def convert_to_tokenized_canonical_smiles(tok_smi):
    """
    Do convertion, if invalid smiles return 'C.C'
    """
    try:
        can_smi = canonicalize(tok_smi.replace(' ','').replace('_', ''))
    except:
        print(tok_smi.replace(' ','').replace('_', ''), '\n')
        return 'C . C'
    return smi_tokenizer(can_smi)


def main(args):
    print(args)

    splitted_path = list(os.path.split(args['input_file']))
    file_name = splitted_path[-1]

    with open(args['input_file'], 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    converted_lines = [convert_to_tokenized_canonical_smiles(line) for line in lines]
    invalid_lines = [x for x in converted_lines if x=='C . C']
    print('number of invalid lines: ', len(invalid_lines))
    print('Size of input file:', len(lines))
    print('Size of output file:', len(converted_lines))
    with open(os.path.join(splitted_path[0], 'can_' + file_name), 'w') as f:
        f.write('\n'.join(converted_lines))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("input_file", help="File containing tokenized smiles.")
    args = parser.parse_args()
    main(vars(args))
   
