import pandas as pd
import os
import numpy as np
import sys
sys.path.append('D:\\CropNutsDocuments\\QC_Model_Predictions')
from predict import predict_chems
from autoencoder import run, denormalize, normalize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import logging
from split import split_spc
from residual_outliers import residual_outliers, residual_outliers_reconstructed, residual_outliers_reconstructed_wetchem
from visualizations import pcc2_confusion_matrix, pcc3_confusion_matrix, pcc1_confusion_matrix



logging.basicConfig(filename='training.log', level=logging.INFO)

chemicals = [
    # 'aluminium',
    # 'phosphorus',
    # 'ph',
    # 'exchangeable_acidity',
    #  'calcium',
    # 'magnesium',
    # 'sulphur', 
    # 'sodium', 
    'iron', 
    # 'manganese',
    # 'boron', 'copper', 'zinc', 'total_nitrogen', 'potassium',
    # 'ec_salts', 'organic_carbon', 'cec',
    #  'sand'
    #   ,'silt', 'clay'
]


data = pd.read_csv('outputFiles/spectra.csv', index_col=0, engine='c')
predict_chems(
    'D:\\CropNutsDocuments\\QC_Model_Predictions\\dl_models_all_chems_20210414\\dl_v2.2_update_2022',
    'outputFiles/predictions',
    chemicals,
    ['DLv2.2'],
    data
    )


residual_outliers(chemicals)

for chem in chemicals:
    split_spc(f"D://CropNutsDocuments/DSML124/outputFiles/PCC1/{chem}.csv", "D://CropNutsDocuments/DSML124/outputFiles/splits", "D://CropNutsDocuments/DSML124/outputFiles/rds", chem)
    # continue
    # if(chem not in os.listdir('outputFiles/models')):
    #     continue
    # if(chem in os.listdir('outputFiles/pcc1_reconstructed_spc')):
    #     continue
    # model = run(chem)
    logging.info(f'Loading model for {chem}')
    model = load_model(f'{os.getcwd()}/outputFiles/models/{chem}')
    spc = pd.read_csv(f'outputFiles/PCC1/test/{chem}.csv', index_col=0)
    outliers_spc = pd.read_csv(f'outputFiles/PCC2/{chem}.csv', index_col=0)
    print(f'outputFiles/models/{chem}')
    logging.info(f'Training for {chem}')

    # model = run(chem)


    spc_no_outliers = np.array(spc)
    spc_pcc2_outliers = np.array(outliers_spc)
    logging.info(f'Converted df to arrays')

    logging.info(f'Normalization of pcc1')
    for i in range(len(spc_no_outliers)):
        spc_no_outliers[i] = normalize(spc_no_outliers[i])

    logging.info(f'Normalization of pcc3')
    for i in range(len(spc_pcc2_outliers)):
        spc_pcc2_outliers[i] = normalize(spc_pcc2_outliers[i])


    logging.info(f'Prediction of pcc1')
    pcc1_predictions = model.predict(spc_no_outliers)
    # if(len(spc_pcc3_outliers) == 0):
    #     continue

    logging.info(f'Prediction of pcc2')
    pcc2_predictions = model.predict(spc_pcc2_outliers)

    logging.info(f'Denormalization of pcc1')
    for i in range(len(pcc1_predictions)):
        min = (np.min(np.array(spc)[i]))
        max = (np.max(np.array(spc)[i]))
        pcc1_predictions[i] = denormalize(pcc1_predictions[i], min, max)

    logging.info(f'Denormalization of pcc3')
    for i in range(len(pcc2_predictions)):
        min = (np.min(np.array(outliers_spc)[i]))
        max = (np.max(np.array(outliers_spc)[i]))
        pcc2_predictions[i] = denormalize(pcc2_predictions[i], min, max)

    pcc1_predictions = pd.DataFrame(pcc1_predictions)
    pcc1_predictions.index = spc.index
    pcc2_predictions = pd.DataFrame(pcc2_predictions)
    pcc2_predictions.index = outliers_spc.index

    logging.info(f'Saving reconstruted spc')
    os.makedirs(f'outputFiles/pcc1_reconstructed_spc', exist_ok=True)
    os.makedirs(f'outputFiles/pcc2_reconstructed_spc', exist_ok=True)
    pcc1_predictions.to_csv(f'outputFiles/pcc1_reconstructed_spc/{chem}.csv')
    pcc2_predictions.to_csv(f'outputFiles/pcc2_reconstructed_spc/{chem}.csv')


for chem in chemicals:
    os.makedirs(f"outputFiles/predictions/pcc1_autoencoded", exist_ok=True)
    os.makedirs(f"outputFiles/predictions/pcc2_autoencoded", exist_ok=True)

    pcc1_reconstructed = pd.read_csv(
        f'outputFiles/pcc1_reconstructed_spc/{chem}.csv', index_col=0)
    pcc3_reconstructed = pd.read_csv(
        f'outputFiles/pcc2_reconstructed_spc/{chem}.csv', index_col=0)

    predict_chems(
        'D:\\CropNutsDocuments\\QC_Model_Predictions\\dl_models_all_chems_20210414\\dl_v2.2_update_2022',
        f'outputFiles/predictions/pcc1_autoencoded',
        [chem],
        ['DLv2.2'],
        pcc1_reconstructed
    )

    predict_chems(
        'D:\\CropNutsDocuments\\QC_Model_Predictions\\dl_models_all_chems_20210414\\dl_v2.2_update_2022',
        f'outputFiles/predictions/pcc2_autoencoded',
        [chem],
        ['DLv2.2'],
        pcc3_reconstructed
    )

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
