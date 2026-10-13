# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *

def whole_gene_expression(list_exp_path, list_condition):
    all_df = pd.DataFrame()
    for exp_path, l_cond in zip(list_exp_path, list_condition):
        for condition in l_cond:
            path = ROOTPATH + exp_path + 'Fish/' + condition +'.csv'
            if not pathlib.Path(path).exists():
                print(f'Skipped (no gene expression table): {path}')
                continue
            gene_expr = pd.read_csv(path, index_col='Key')
            gene_expr.index.name = 'fish'

            gene_expr['condition'] = condition
            gene_expr['exp_path'] = exp_path
            gene_expr = gene_expr.set_index(['exp_path', 'condition'], append=True)
            all_df = pd.concat((all_df, gene_expr))
    return(all_df)

# Create a single big file
gene_list = ["CORO1A", "TNF", "ACTB", "IL2RA", "IRF8", "CD69", "CCR7", "GZMB", "GBP4", "IFNG", "XP01", "TNF_2", "CD8"] # , "CORO1A_2"
list_exp_path, list_condition = LEARNING_EXP_PATHS, LEARNING_CONDITIONS_GENE_PROFILE

all_genes =  whole_gene_expression(list_exp_path, list_condition)
# Do not replace the table by an empty one
if len(all_genes) == 0:
    raise FileNotFoundError('No gene expression table found for the experiments and conditions of config.yaml: 6_results/all_gene_profile.csv is not modified')
all_genes.to_csv('6_results/all_gene_profile.csv', index=True)
