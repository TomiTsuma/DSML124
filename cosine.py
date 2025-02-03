from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def getSimilarityMatrix(type="PCC1", chem="organic_carbon"):
    if type == "PCC1":
        actual_spc = pd.read_csv(f"outputFiles/PCC1/test/{chem}.csv", index_col=0)
        reconstructed_spc = pd.read_csv(f"outputFiles/pcc1_reconstructed_spc/{chem}.csv", index_col=0)
    if type == "PCC3":
        actual_spc = pd.read_csv(f"outputFiles/PCC3/{chem}.csv", index_col=0)
        reconstructed_spc = pd.read_csv(f"outputFiles/pcc3_reconstructed_spc/{chem}.csv", index_col=0)