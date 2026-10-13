'''
Print the parameters of config.yaml as shell variables, for the job.slurm files and the manual steps
Use: eval "$(python3 -m tools.config_to_shell)"
'''
import contextlib
import io
import shlex

# Do not print the messages of read_yaml: the output is evaluated by the shell
with contextlib.redirect_stdout(io.StringIO()):
    from tools.read_yaml import *

variables = {
    'ROOTPATH': ROOTPATH,
    'EXP_PATH': EXP_PATH,
    'CONDITION': CONDITION,
    'LIVE': ROOTPATH + EXP_PATH + 'Live/' + CONDITION,
    'FISH': ROOTPATH + EXP_PATH + 'Fish',
    'MODEL_FINETUNE': MODEL_FINETUNE_PATH,
    'N_FRAMES': N_FRAMES,
    'Z_FINETUNE': str(Z_FINETUNE).zfill(2),
    'FISH_N_TILES': FISH_N_TILES,
}

for name, value in variables.items():
    print(f'{name}={shlex.quote(str(value))}')
