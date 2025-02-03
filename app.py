import pandas as pd
import os
import numpy as np
import sys
import torch
import pickle
import pickle
sys.path.append('/mnt/batch/tasks/shared/LS_root/mounts/clusters/cnls-ds-compute-instance/code/Users/tsuma.thomas/DSML124/QC_Model_Predictions')
from predict import predict_chems
from autoencoder import run, denormalize, normalize, run_lstm, RecurrentAutoencoder, run_lstm_autoencoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import logging
from split import split_spc
from residual_outliers import residual_outliers, residual_outliers_reconstructed, residual_outliers_reconstructed_wetchem

# from visualizations import pcc2_confusion_matrix, pcc3_confusion_matrix, pcc1_confusion_matrix


logging.basicConfig(filename='training.log', level=logging.INFO)

chemicals = [
    # 'aluminium'
    # 'phosphorus',
    # 'ph',
    # 'exchangeable_acidity',
    #  'calcium',
    # 'magnesium',
    # 'sulphur', 
    # 'sodium',
    # 'iron', 
    # 'manganese',
    # 'boron', 
    # 'copper', 
    # 'zinc', 
    # 'total_nitrogen', 
    # 'potassium',
    # 'ec_salts', 
    'organic_carbon'
    # , 'cec',
    #  'sand'
    #   ,'silt', 'clay'
]


data = pd.read_csv('outputFiles/v1_spectra.csv', index_col=0, engine='c')

predict_chems(
    '/mnt/batch/tasks/shared/LS_root/mounts/clusters/cnls-ds-compute-instance/code/Users/tsuma.thomas/QC_Model_Predictions/dl_models_all_chems_20210414/v2.2',
    '/mnt/batch/tasks/shared/LS_root/mounts/clusters/cnls-ds-compute-instance/code/Users/tsuma.thomas/DSML124/outputFiles/predictions',
    chemicals,
    ['DLv2.2'],
    data
    )


# residual_outliers(chemicals)

for chem in chemicals:
    print(chem)

    # split_spc(f"{os.getcwd()}/outputFiles/v1_spectra.csv", f"{os.getcwd()}/outputFiles/rds", f"{os.getcwd()}/outputFiles/splits", chem)
    # model = run(chem)
    model = load_model(f'{os.getcwd()}/outputFiles/models/{chem}')



    spc = pd.read_csv(f'outputFiles/PCC1/test/{chem}.csv', index_col=0)

    outliers_spc = pd.DataFrame()
    if(f"{chem}.csv" in os.listdir("outputFiles/PCC2")):
        _ = pd.read_csv(f'outputFiles/PCC2/{chem}.csv', index_col=0)
        outliers_spc = pd.concat([outliers_spc,_])
    if(f"{chem}.csv" in os.listdir("outputFiles/PCC3")):
        _ = pd.read_csv(f'outputFiles/PCC3/{chem}.csv', index_col=0)
        outliers_spc = pd.concat([outliers_spc,_])
    logging.info(f'Training for {chem}')



    logging.info(f'Normalization of pcc1')
    scaler = pickle.load(open("/mnt/batch/tasks/shared/LS_root/mounts/clusters/cnls-ds-compute-instance/code/Users/tsuma.thomas/DSML124/outputFiles/scalers/{}.pkl".format(chem), "rb"))

    spc_no_outliers = scaler.transform(spc)

    logging.info(f'Normalization of pcc2')
    spc_pcc2_outliers = scaler.transform(outliers_spc)


    logging.info(f'Prediction of pcc1')
    print(spc_no_outliers)
    pcc1_predictions = model.predict(spc_no_outliers)

    logging.info(f'Prediction of pcc2')
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
    os.makedirs(f'outputFiles/pcc2_reconstructed_spc', exist_ok=True)
    pcc1_predictions.to_csv(f'outputFiles/pcc1_reconstructed_spc/{chem}.csv')
    pcc2_predictions.to_csv(f'outputFiles/pcc2_reconstructed_spc/{chem}.csv')


# for chem in chemicals:
#     print(chem)
#     os.makedirs(f"outputFiles/predictions/pcc1_autoencoded", exist_ok=True)
#     os.makedirs(f"outputFiles/predictions/pcc2_autoencoded", exist_ok=True)

#     pcc1_reconstructed = pd.read_csv(
#         f'outputFiles/pcc1_reconstructed_spc/{chem}.csv', index_col=0)
#     pcc3_reconstructed = pd.read_csv(
#         f'outputFiles/pcc2_reconstructed_spc/{chem}.csv', index_col=0)

#     print(len(pcc1_reconstructed))
#     print(len(pcc3_reconstructed))

#     predict_chems(
#         '/mnt/batch/tasks/shared/LS_root/mounts/clusters/cnls-ds-compute-instance/code/Users/tsuma.thomas/QC_Model_Predictions/dl_models_all_chems_20210414/v2.2',
#         f'outputFiles/predictions/pcc1_autoencoded',
#         [chem],
#         ['DLv2.2'],
#         pcc1_reconstructed
#     )

#     predict_chems(
#         '/mnt/batch/tasks/shared/LS_root/mounts/clusters/cnls-ds-compute-instance/code/Users/tsuma.thomas/QC_Model_Predictions/dl_models_all_chems_20210414/v2.2',
#         f'outputFiles/predictions/pcc2_autoencoded',
#         [chem],
#         ['DLv2.2'],
#         pcc3_reconstructed
#     )

residual_outliers_reconstructed(chemicals)
residual_outliers_reconstructed_wetchem(chemicals)

# for chem in chemicals:
    # pcc2_confusion_matrix(chem)
    # pcc3_confusion_matrix(chem)
    # pcc1_confusion_matrix(chem)


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
