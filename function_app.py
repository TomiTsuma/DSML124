import azure.functions as func
import json
import logging
import tensorflow
import pickle
# import sys
import pandas as pd
# sys.path.append('./QC_Model_Predictions')
from QC_Model_Predictions.predict import predict_chems
app = func.FunctionApp()

@app.route(route="spectral-outliers", auth_level=func.AuthLevel.ANONYMOUS)
def spetral_outliers(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    difference_thresholds = pickle.load(open('thresholds.dict','rb'))

    chemicals = [i for i in  difference_thresholds.keys()]

    body = req.get_json()                                                                                                           
    result_df = pd.DataFrame()
    original_spc = pd.DataFrame(body).T
    original_predictions = predict_chems(
    './QC_Model_Predictions/dl_models_all_chems_20210414/v2.2',
    './outputFiles/predictions/test_preds',
    chemicals,
    ['DLv2.2'],
    original_spc
    )

    result_df.index = original_spc.index

    result_dict = {}

    for chem in chemicals:
        scaler = pickle.load(open("./outputFiles/scalers/{}.pkl".format(chem), "rb"))

        normalized_spc =scaler.transform(original_spc)
        model = tensorflow.keras.models.load_model(f"./outputFiles/models/{chem}")
        reconstructed_spc = model.predict(normalized_spc)
        reconstructed_spc = scaler.inverse_transform(reconstructed_spc)

        reconstructed_spc = pd.DataFrame(reconstructed_spc)
        reconstructed_spc.index = original_spc.index

        reconstructed_predictions = predict_chems(
        './QC_Model_Predictions/dl_models_all_chems_20210414/v2.2',
        './outputFiles/predictions/test_preds',
        [chem],
        ['DLv2.2'],
        reconstructed_spc
        )


        result_df[f'{chem}_difference'] = abs(reconstructed_predictions[chem] - original_predictions[chem])

        result_df.loc[result_df[f'{chem}_difference'] > difference_thresholds[chem], chem] = True
        result_df.loc[result_df[f'{chem}_difference'] <= difference_thresholds[chem], chem] = False
    result_dict = result_df.T.to_dict()

    return func.HttpResponse(json.dumps(result_dict) , status_code=200)