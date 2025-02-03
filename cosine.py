from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os


def getSimilarityMatrix(type="PCC1", chem="organic_carbon"):
    os.makedirs(f"outputFiles/similarity_matrix/{type}", exist_ok=True) 
    if type == "PCC1":
        actual_spc = pd.read_csv(f"outputFiles/PCC1/test/{chem}.csv", index_col=0)
        reconstructed_spc = pd.read_csv(f"outputFiles/pcc1_reconstructed_spc/{chem}.csv", index_col=0)
        similarity_matrix = cosine_similarity(actual_spc, reconstructed_spc)
    if type == "PCC3":
        actual_spc = pd.read_csv(f"outputFiles/PCC3/{chem}.csv", index_col=0)
        reconstructed_spc = pd.read_csv(f"outputFiles/pcc3_reconstructed_spc/{chem}.csv", index_col=0)
        similarity_matrix = cosine_similarity(actual_spc, reconstructed_spc)
    similarity_df = pd.DataFrame(similarity_matrix, index=actual_spc.index, columns=reconstructed_spc.index)
    similarity_df.to_csv(f"outputFiles/similarity_matrix/{type}/{chem}.csv")
