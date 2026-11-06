'''
Step 5.2: table of the gene expression of all the smFISH cells of the dataset (learning.exp_paths and
learning.conditions_gene_profile), matched to a live cell or not.

Input:  ROOTPATH/{exp_path}/Fish/{condition}.csv (step 2.6)
Output: exp/results/all_gene_profile.csv (index: fish, exp_path, condition)

Run: python3 -m 5_POC_activation.2_concatenate_gene_expr
'''


# Import local tools
import pandas as pd
import pathlib
from tools.read_yaml import LEARNING_CONDITIONS_GENE_PROFILE, LEARNING_EXP_PATHS, RESULTS_PATH, ROOTPATH


def whole_gene_expression(list_exp_path, list_condition):
    all_df = pd.DataFrame()
    for exp_path, l_cond in zip(list_exp_path, list_condition):
        for condition in l_cond:
            path = ROOTPATH + exp_path + 'Fish/' + condition + '.csv'
            if not pathlib.Path(path).exists():
                print(f'Skipped (no gene expression table): {path}')
                continue
            gene_expr = pd.read_csv(path, index_col='Key')
            gene_expr.index.name = 'fish'

            gene_expr['condition'] = condition
            gene_expr['exp_path'] = exp_path
            gene_expr = gene_expr.set_index(['exp_path', 'condition'], append=True)
            all_df = pd.concat((all_df, gene_expr))
    return (all_df)


def main():
    '''Build and save the table of all the gene profiles'''
    # Create a single big file
    list_exp_path, list_condition = LEARNING_EXP_PATHS, LEARNING_CONDITIONS_GENE_PROFILE

    all_genes = whole_gene_expression(list_exp_path, list_condition)
    # Do not replace the table by an empty one
    if len(all_genes) == 0:
        raise FileNotFoundError('No gene expression table found for the experiments and conditions of config.yaml: exp/results/all_gene_profile.csv is not modified')
    all_genes.to_csv(RESULTS_PATH + 'all_gene_profile.csv', index=True)


if __name__ == "__main__":
    main()
