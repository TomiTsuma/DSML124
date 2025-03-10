import pandas as pd
import os
import notebook
from pathlib import Path
import subprocess
# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri

# r = robjects.r
# r['source']('splits.R')
# split = robjects.globalenv['split']

try:
    os.makedirs("./outputFiles/splits", exist_ok=True)
    os.makedirs("./outputFiles/rds", exist_ok=True)
except Exception as e:
    print(e)


# pandas2ri.activate()


def split_spc(spc_path, rds_path, splits_path, chem, nrows=None):
    os.makedirs(f"outputFiles/PCC1/train", exist_ok=True)
    os.makedirs(f"outputFiles/PCC1/test", exist_ok=True)
    os.makedirs(f"outputFiles/PCC1/validation", exist_ok=True)
    # subprocess.run(['sudo', 'Rscript', '/home/tom/DSML124/splits.r', f'{chem}'])
    # split(spc_path, rds_path, splits_path, chem)
    spc = pd.read_csv(spc_path, index_col=0, engine='c')

    train = pd.read_csv(f"{splits_path}/{chem}_train_sample_codes.csv")
    test = pd.read_csv(f"{splits_path}/{chem}_test_sample_codes.csv")
    valid = pd.read_csv(f"{splits_path}/{chem}_valid_sample_codes.csv")

    train = spc.loc[spc.index.isin(train['x'])]
    test = spc.loc[spc.index.isin(test['x'])]
    valid = spc.loc[spc.index.isin(valid['x'])]

    assert len(train) > 0
    assert len(test) > 0
    assert len(valid) > 0
    train.to_csv(f"./outputFiles/PCC1/train/{chem}.csv")
    test.to_csv(f"./outputFiles/PCC1/test/{chem}.csv")
    valid.to_csv(f"./outputFiles/PCC1/validation/{chem}.csv")

