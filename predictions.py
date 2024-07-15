import pandas as pd
import os
import sys
sys.path.append('D:\\CropNutsDocuments\\QC_Model_Predictions')
from predict import predict_chems
import logging
logging.basicConfig(filename='training.log', level=logging.INFO)


chemicals = ['aluminium', 
            'phosphorus', 'ph', 'exchangeable_acidity', 'calcium', 'magnesium',
              'sulphur', 'sodium', 'iron', 'manganese', 'boron', 'copper', 'zinc', 'total_nitrogen', 'potassium',
             'ec_salts', 'organic_carbon', 'cec', 'sand', 'silt', 'clay']
for chem in chemicals:
    if(f'{chem}.csv' not in os.listdir('outputFiles/pcc1_reconstructed_spc')):
        continue
    pcc1_predictions = pd.read_csv(f'outputFiles/pcc1_reconstructed_spc/{chem}.csv', index_col=0)
    pcc3_predictions = pd.read_csv(f'outputFiles/pcc3_reconstructed_spc/{chem}.csv', index_col=0)
    os.makedirs(f'outputFiles/pcc1_reconstructed_predictions', exist_ok=True)
    os.makedirs(f'outputFiles/pcc3_reconstructed_predictions', exist_ok=True)
    logging.info(f'Making predictions of pcc1 {chem}')
    predict_chems(
      'D:\\CropNutsDocuments\\QC_Model_Predictions\\dl_models_all_chems_20210414\\dl_v2.2_update_2022', 
      'outputFiles/pcc1_reconstructed_predictions', 
      [chem], 
      ['DLv2.2'], 
      pcc1_predictions
    )

    logging.info(f'Making predictions of pcc3 {chem}')
    predict_chems(
      'D:\\CropNutsDocuments\\QC_Model_Predictions\\dl_models_all_chems_20210414\\dl_v2.2_update_2022', 
      'outputFiles/pcc3_reconstructed_predictions', 
      [chem], 
      ['DLv2.2'], 
      pcc3_predictions
    )
