import tensorflow

import psycopg2
import pandas as pd
import ast
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path
import glob
import pyodbc
from datetime import datetime, timedelta
import pickle
import logging
from QC_Model_Predictions.predict import predict_chems
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


logging.basicConfig(filename='autocheck.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_cursor():
    username = "doadmin"
    password = 'yzmodwh2oh16iks6'
    host = 'db-postgresql-cl1-do-user-2276924-0.db.ondigitalocean.com'
    port = 25060
    database = 'MandatoryMetadata'
    schema = 'historical'

    conn = psycopg2.connect(host=host, database=database,
                            user=username, password=password, port=port)
    cur = conn.cursor()
    cur.execute("SET search_path TO " + schema)

    return conn, cur
conn, cur = get_db_cursor()

def convertSpectra(df):
    print("Converting spectra")
    df_ = pd.DataFrame([i[[i for i in i.keys()][0]] for i in df['averaged_spectra'].values],columns = np.arange(522,3977,2))
    df_.index = df.index
    print("Spectra converted")
    return df_

def getSpectralCodes():
    yesterday = datetime.now() - timedelta(1)
    yesterday = yesterday.strftime('%Y-%m-%d')
    query = f"""
    SELECT DISTINCT mandatorymetadata.sample_code  
    FROM spectraldata 
    INNER JOIN mandatorymetadata 
    ON mandatorymetadata.metadata_id = spectraldata.metadata_id 
    WHERE 
    is_finalized=True AND 
    passed=True AND 
    is_active=True AND 
    averaged=True AND 
    mandatorymetadata.sensor_id = 1 AND  
    mandatorymetadata.sample_type_id = 1 AND
    mandatorymetadata.sample_pretreatment_id = 1 AND
    mandatorymetadata.timestamp > {yesterday}
    LIMIT 2
    """

    return pd.read_sql(query, con=conn)


def get_spc():
    
    sample_codes = getSpectralCodes()['sample_code'].tolist()
    spectra = pd.DataFrame(columns=['sample_code','averaged_spectra'])

    if(len(sample_codes) < 5000):
        count  = len(sample_codes)
        step=count
    elif(len(sample_codes) < 70000):
        count = len(sample_codes)
        step=5000
    else:
        count = len(sample_codes)
        step=5000
    start = 0



    for i in np.arange(start, count, step):
        
        print("Fetching spectra from {}".format(start))
        samples = [i for i in sample_codes][start:start+step]
        query = f"""
        SELECT spectraldata.metadata_id, averaged_spectra, mandatorymetadata.sample_code  
        FROM spectraldata 
        INNER JOIN mandatorymetadata 
        ON mandatorymetadata.metadata_id = spectraldata.metadata_id 
        WHERE 
        is_finalized=True AND 
        passed=True AND 
        is_active=True AND 
        averaged=True AND 
        mandatorymetadata.sensor_id = 1 AND  
        mandatorymetadata.sample_type_id = 1 AND
        mandatorymetadata.sample_pretreatment_id = 1 AND
        sample_code IN {str(samples).replace('[','(').replace(']',')')}"""

        _ = pd.read_sql(query, con=conn)
        spectra = pd.concat([spectra, _], axis=0)
        start = start + step
        if (count-step) > 5000:
            step=5000
        else:
            step = count-step

    conn.close()
    spectra = spectra[['sample_code', 'averaged_spectra']]
    spectra = spectra.set_index('sample_code')
    spectra = convertSpectra(spectra)

spc = get_spc()

def dataframe_to_dict(df):
    return df.apply(lambda row: row.tolist(), axis=1).to_dict()

spectra_dict = dataframe_to_dict(spc)
print(spectra_dict)

def spetral_outliers(req):
    try:
        difference_thresholds = pickle.load(open('thresholds.dict','rb'))

        chemicals = ["organic_carbon", "total_nitrogen"]

        body = req.get_json()                                                                                                           
        result_df = pd.DataFrame()
        original_spc = pd.DataFrame(body).T
        original_predictions = predict_chems(
            './QC_Model_Predictions/dl_models_all_chems_20210414/v2.2',
            './outputFiles/predictions/test_preds',
            chemicals,
            ['v2.3'],
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

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name('./credentials.json', scope)
        client = gspread.authorize(creds)

        spreadsheet = client.create(f"Spectral Outliers ({str(datetime.now().strftime('%Y-%m-%d %H-%M-%S'))})")
        sheet = spreadsheet.get_worksheet(0)
        sheet.update([result_df.columns.values.tolist()] + result_df.values.tolist())

        spreadsheet.share(None, perm_type='anyone', role='reader')
        spreadsheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}"
        print(spreadsheet_url)

        

        # Your email credentials
        EMAIL_ADDRESS = "tsuma.thomas@cropnuts.com"
        EMAIL_PASSWORD = "yirh yqja wqwr hhxp"

        # Email details
        receiver_email = "tsuma.thomas@cropnuts.com"
        subject = "Spectral Outliners"
        body = f"Here is the link to the spectral outliers: {spreadsheet_url}"

        # Create email message
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Send email via Gmail SMTP server
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()  # Secure connection
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.sendmail(EMAIL_ADDRESS, receiver_email, msg.as_string())
                print("Email sent successfully!")
        except Exception as e:
            print(f"Error: {e}")

    except Exception as e:
        logging.error(f"Error in spetral_outliers: {e}")
        result_dict = {"error": str(e)}


spetral_outliers(spectra_dict)