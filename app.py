import pandas as pd
import os
import numpy as np
import sys
import torch
import pickle
import subprocess
sys.path.append('/home/tom/DSML124/QC_Model_Predictions')
from predict import predict_chems
from autoencoder import run, denormalize, normalize, run_lstm, RecurrentAutoencoder, run_lstm_autoencoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import logging
from split import split_spc
from residual_outliers import residual_outliers, residual_outliers_reconstructed, residual_outliers_reconstructed_wetchem
from datetime import datetime
from cosine import getSimilarityMatrix
from data import get_spc, getWetchem
# from visualizations import pcc2_confusion_matrix, pcc3_confusion_matrix, pcc1_confusion_matrix


logging.basicConfig(filename='training.log', level=logging.INFO)
class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

formatter = CustomFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.FileHandler('training.log')
handler.setFormatter(formatter)
logging.getLogger().handlers = [handler]


chemicals = [
    'aluminium',
    'phosphorus',
    'ph',
    'exchangeable_acidity',
     'calcium',
    'magnesium',
    'sulphur', 
    'sodium',
    'iron', 
    'manganese',
    'boron', 
    'copper', 
    'zinc', 
    # 'total_nitrogen', 
    'potassium',
    'ec_salts', 
    # 'organic_carbon', 
    'cec',
    'sand',
    'silt', 'clay'
]
chemicals = [ i for i in chemicals if i in os.listdir("/home/tom/DSML124/QC_Model_Predictions/dl_models_all_chems_20210414/v2.3")]
# get_spc()
# getWetchem(chemicals)
# wetchem = pd.read_csv("outputFiles/cleaned_wetchem.csv")
# wetchem = wetchem.set_index("sample_code")
# data = pd.read_csv('outputFiles/spectra.csv', index_col=0, engine='c')

# data = data.loc[data.index.isin(wetchem.index)]

# predict_chems(
#     '/home/tom/DSML124/QC_Model_Predictions/dl_models_all_chems_20210414/v2.3',
#     '/home/tom/DSML124/outputFiles/predictions',
#     chemicals,
#     ['v2.3'],
#     data
#     )


# residual_outliers(chemicals, "v2.3")
# for chem in chemicals:
#     spc_tmp = pd.read_csv(f"{os.getcwd()}/outputFiles/PCC1/{chem}.csv",engine='c',index_col=0)
#     spc_tmp = spc_tmp.set_index([ i.strip() for i in spc_tmp.index ])
#     print(spc_tmp.index[0:5])
#     print("Len b4 dropping duplicates",len(spc_tmp))
#     spc_tmp = spc_tmp[~spc_tmp.index.duplicated(keep='first')]
#     print("Len after dropping duplicates",len(spc_tmp))
#     spc_tmp.to_csv(f"{os.getcwd()}/outputFiles/PCC1/{chem}.csv")


# subprocess.run(["sudo","bash","/home/tom/DSML124/split_data.sh"])

for chem in chemicals:

    print(chem)

    split_spc(f"{os.getcwd()}/outputFiles/PCC1/{chem}.csv", f"{os.getcwd()}/outputFiles/rds", f"{os.getcwd()}/outputFiles/splits", chem)
    logging.info(f'Training for {chem}')
    model = run(chem)
    model = load_model(f'{os.getcwd()}/outputFiles/models/{chem}')



    spc = pd.read_csv(f'outputFiles/PCC1/test/{chem}.csv', index_col=0)

    outliers_spc = pd.DataFrame()
    if(f"{chem}.csv" in os.listdir("outputFiles/PCC2")):
        _ = pd.read_csv(f'outputFiles/PCC2/{chem}.csv', index_col=0)
        outliers_spc = pd.concat([outliers_spc,_])
    if(f"{chem}.csv" in os.listdir("outputFiles/PCC3")):
        _ = pd.read_csv(f'outputFiles/PCC3/{chem}.csv', index_col=0)
        outliers_spc = pd.concat([outliers_spc,_])
    



    logging.info(f'Normalization of pcc1')
    scaler = pickle.load(open("/home/tom/DSML124/outputFiles/scalers/{}.pkl".format(chem), "rb"))

    spc_no_outliers = scaler.transform(spc)

    logging.info(f'Normalization of pcc3')
    spc_pcc2_outliers = scaler.transform(outliers_spc)


    logging.info(f'Prediction of pcc1')
    print(spc_no_outliers)
    pcc1_predictions = model.predict(spc_no_outliers)

    logging.info(f'Prediction of pcc3')
    pcc2_predictions = model.predict(spc_pcc2_outliers)

    logging.info(f'Denormalization of pcc1')
    pcc1_predictions = scaler.inverse_transform(pcc1_predictions)

    logging.info(f'Denormalization of pcc3')
    pcc2_predictions = scaler.inverse_transform(pcc2_predictions)

    pcc1_predictions = pd.DataFrame((pcc1_predictions))
    pcc1_predictions.index = spc.index
    pcc2_predictions = pd.DataFrame((pcc2_predictions))
    pcc2_predictions.index = outliers_spc.index

    logging.info(f'Saving reconstruted spc')
    print(len(pcc1_predictions))
    print(len(pcc2_predictions))
    os.makedirs(f'outputFiles/pcc1_reconstructed_spc', exist_ok=True)
    os.makedirs(f'outputFiles/pcc3_reconstructed_spc', exist_ok=True)
    pcc1_predictions.to_csv(f'outputFiles/pcc1_reconstructed_spc/{chem}.csv')
    pcc2_predictions.to_csv(f'outputFiles/pcc3_reconstructed_spc/{chem}.csv')


# for chem in chemicals:
#     print(chem)
#     os.makedirs(f"outputFiles/predictions/pcc1_autoencoded", exist_ok=True)
#     os.makedirs(f"outputFiles/predictions/pcc3_autoencoded", exist_ok=True)

#     pcc1_reconstructed = pd.read_csv(
#         f'outputFiles/pcc1_reconstructed_spc/{chem}.csv', index_col=0)
#     pcc3_reconstructed = pd.read_csv(
#         f'outputFiles/pcc3_reconstructed_spc/{chem}.csv', index_col=0)

#     print(len(pcc1_reconstructed))
#     print(len(pcc3_reconstructed))

#     predict_chems(
#         '/home/tom/DSML124/QC_Model_Predictions/dl_models_all_chems_20210414/v2.3',
#         f'outputFiles/predictions/pcc1_autoencoded',
#         [chem],
#         ['v2.3'],
#         pcc1_reconstructed
#     )

#     predict_chems(
#         '/home/tom/DSML124/QC_Model_Predictions/dl_models_all_chems_20210414/v2.3',
#         f'outputFiles/predictions/pcc2_autoencoded',
#         [chem],
#         ['v2.3'],
#         pcc3_reconstructed
#     )
# for chem in chemicals:
#     getSimilarityMatrix(type="PCC1", chem=chem)
#     getSimilarityMatrix(type="PCC3", chem=chem)

# residual_outliers_reconstructed(chemicals)
# residual_outliers_reconstructed_wetchem(chemicals)

# # for chem in chemicals:
# #     pcc2_confusion_matrix(chem)
# #     pcc3_confusion_matrix(chem)
# #     pcc1_confusion_matrix(chem)


# for chem in chemicals:
#     os.makedirs(f"outputFiles/visualizations/{chem}", exist_ok=True)
#     df = pd.read_csv(f"outputFiles/PCC1_Classes_Reconstructed/{chem}.csv")
#     df = df[[chem, '0']]

#     plt.scatter(df[chem], df['0'])
#     plt.savefig(f"outputFiles/visualizations/{chem}/PCC1_Reconstructed_vs_Original.png")
#     plt.clf()

#     df = pd.read_csv(f"outputFiles/PCC3_Classes_Reconstructed/{chem}.csv")
#     df = df[[chem, '0']]

#     plt.scatter(df[chem], df['0'])
#     plt.savefig(f"outputFiles/visualizations/{chem}/PCC3_Reconstructed_vs_Original.png")
#     plt.clf()

#     df = pd.read_csv(f"outputFiles/PCC_Classes/{chem}.csv")
#     df = df[[chem, '0']]

#     plt.scatter(df[chem], df['0'])
#     plt.savefig(f"outputFiles/visualizations/{chem}/Wetchem_vs_Original.png")
#     plt.clf()

#     df = pd.read_csv(f"outputFiles/PCC1_Wetchem_Reconstructed/{chem}.csv")
#     df = df[[chem, '0']]

#     plt.scatter(df[chem], df['0'])
#     plt.savefig(f"outputFiles/visualizations/{chem}/PCC1_Wetchem_vs_Reconstructed.png")
#     plt.clf()

#     df = pd.read_csv(f"outputFiles/PCC3_Wetchem_Reconstructed/{chem}.csv")
#     df = df[[chem, '0']]

#     plt.scatter(df[chem], df['0'])
#     plt.savefig(f"outputFiles/visualizations/{chem}/PCC3_Wetchem_vs_Reconstructed.png")
#     plt.clf()
