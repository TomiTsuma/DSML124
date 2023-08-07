import pandas as pd
import os
import numpy as np
import sys
sys.path.append('D:\\CropNutsDocuments\\QC_Model_Predictions')
from predict import predict_chems
from autoencoder import run, denormalize, normalize
import matplotlib.pyplot as plt
import seaborn as sns
from residual_outliers import residual_outliers, residual_outliers_reconstructed
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import logging
logging.basicConfig(filename='training.log', level=logging.INFO)

chemicals = [
    # 'aluminium', 
            # 'phosphorus', 
            'ph',
              'exchangeable_acidity', 'calcium', 'magnesium',
              'sulphur', 'sodium', 'iron', 'manganese', 'boron', 'copper', 'zinc', 'total_nitrogen', 'potassium',
             'ec_salts', 'organic_carbon', 'cec', 'sand', 'silt', 'clay'
            ]

             
# data = pd.read_csv('outputFiles/spectra.csv', index_col=0, engine='c')
# predict_chems(
#     'D:\\CropNutsDocuments\\QC_Model_Predictions\\dl_models_all_chems_20210414\\dl_v2.2_update_2022', 
#     'outputFiles/predictions', 
#     chemicals, 
#     ['DLv2.2'], 
#     data
#     )


# residual_outliers()

for chem in chemicals:
    # if(chem not in os.listdir('outputFiles/models')):
    #     continue
    # if(chem in os.listdir('outputFiles/pcc1_reconstructed_spc')):
    #     continue
    model = run(chem)
    logging.info(f'Loading model for {chem}')
    model = load_model(f'{os.getcwd()}/outputFiles/models/{chem}')
    spc = pd.read_csv(f'outputFiles/PCC1/{chem}.csv', index_col=0)
    outliers_spc = pd.read_csv(f'outputFiles/PCC3/{chem}.csv', index_col=0)
    print(f'outputFiles/models/{chem}')
    logging.info(f'Training for {chem}')

    # model = run(chem)
    
    
    spc_no_outliers = np.array(spc)
    spc_pcc3_outliers = np.array(outliers_spc)
    logging.info(f'Converted df to arrays')

    logging.info(f'Normalization of pcc1')
    for i in range(len(spc_no_outliers)):
        spc_no_outliers[i] = normalize(spc_no_outliers[i])

    logging.info(f'Normalization of pcc3')
    for i in range(len(spc_pcc3_outliers)):
        spc_pcc3_outliers[i] = normalize(spc_pcc3_outliers[i])
    

    logging.info(f'Prediction of pcc1')
    pcc1_predictions = model.predict(spc_no_outliers)
    if(len(spc_pcc3_outliers) == 0):
        continue
  
    logging.info(f'Prediction of pcc3')
    pcc3_predictions = model.predict(spc_pcc3_outliers)

    logging.info(f'Denormalization of pcc1')
    for i in range(len(pcc1_predictions)):
        min = (np.min(np.array(spc)[i]))
        max = (np.max(np.array(spc)[i]))
        pcc1_predictions[i] = denormalize(pcc1_predictions[i], min, max)

    logging.info(f'Denormalization of pcc3')
    for i in range(len(pcc3_predictions)):
        min = (np.min(np.array(outliers_spc)[i]))
        max = (np.max(np.array(outliers_spc)[i]))
        pcc3_predictions[i] = denormalize(pcc3_predictions[i], min, max)


    pcc1_predictions = pd.DataFrame(pcc1_predictions)
    pcc1_predictions.index = spc.index
    pcc3_predictions = pd.DataFrame(pcc3_predictions)
    pcc3_predictions.index = outliers_spc.index

    logging.info(f'Saving reconstruted spc')
    os.makedirs(f'outputFiles/pcc1_reconstructed_spc', exist_ok=True)
    os.makedirs(f'outputFiles/pcc3_reconstructed_spc', exist_ok=True)
    pcc1_predictions.to_csv(f'outputFiles/pcc1_reconstructed_spc/{chem}.csv')
    pcc3_predictions.to_csv(f'outputFiles/pcc3_reconstructed_spc/{chem}.csv')



for chem in chemicals:
    os.makedirs(f"outputFiles/predictions/pcc1_autoencoded", exist_ok=True)
    os.makedirs(f"outputFiles/predictions/pcc3_autoencoded", exist_ok=True)

    pcc1_reconstructed = pd.read_csv(f'outputFiles/pcc1_reconstructed_spc/{chem}.csv', index_col=0)
    pcc3_reconstructed = pd.read_csv(f'outputFiles/pcc3_reconstructed_spc/{chem}.csv', index_col=0)

    predict_chems(
    'D:\\CropNutsDocuments\\QC_Model_Predictions\\dl_models_all_chems_20210414\\dl_v2.2_update_2022', 
    f'outputFiles/predictions/pcc1_autoencoded', 
    [chem], 
    ['DLv2.2'], 
    pcc1_reconstructed
    )

    predict_chems(
    'D:\\CropNutsDocuments\\QC_Model_Predictions\\dl_models_all_chems_20210414\\dl_v2.2_update_2022', 
    f'outputFiles/predictions/pcc3_autoencoded', 
    [chem], 
    ['DLv2.2'], 
    pcc3_reconstructed
    )
