import psycopg2
import pandas as pd
import ast
import numpy as np
import os
from dotenv import load_dotenv

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
    mandatorymetadata.sample_pretreatment_id = 1"""

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
    spectra.to_csv('outputFiles/spectra.csv')
get_spc()

def load_residual_outliers():
    redbooth_outliers = {'boron' : [-5, 5], 'phosphorus' : [-250, 450],'zinc' : [-25, 100], 'sulphur' : [-100, 400],'sodium':[-1000,2500], 'magnesium':[-500,1000],'potassium' : [-800,1600],'calcium':[-5000,5000],'copper' : [-100,300],'ec_salts' : [-1000,2000],'organic_carbon' : [-2,2] }
    redbooth_properties = [i for i in redbooth_outliers.keys()]

    pcc_classes_dict = pd.read_csv("element_managemet_thresholds.csv", index_col=0).T.to_dict()
    pcc_elements = [i for i in pcc_classes_dict.keys() if i not in redbooth_properties]
    pcc_classes_dict = {key: pcc_classes_dict[key] for key in pcc_classes_dict.keys() if key in pcc_elements}


    all_chemicals = ['aluminium', 
            'phosphorus', 'ph', 'exchangeable_acidity', 'calcium', 'magnesium',
              'sulphur', 'sodium', 'iron', 'manganese', 'boron', 'copper', 'zinc', 'total_nitrogen', 'potassium',
             'ec_salts', 'organic_carbon', 'cec', 'sand', 'silt', 'clay']

    undefined_chems = [i for i in all_chemicals if (i not in  pcc_elements and i not in redbooth_properties)]

    uncleaned_wetchem_df = pd.read_csv("inputFiles/cleaned_wetchem.csv")
    uncleaned_wetchem_df = uncleaned_wetchem_df.rename(columns={"Unnamed: 0":"sample_code"})
    uncleaned_wetchem_df.set_index("sample_code")
    for column in uncleaned_wetchem_df.columns:
        if(column != 'sample_code'):
            vals = []
            for value in uncleaned_wetchem_df[column].values:
                if(value is not None):
                    value = str(value)
                    value = value.replace(">","").replace("<","").replace("...","").strip()
                    value = float(value)
                vals.append(value)
            uncleaned_wetchem_df[column] = vals
    wetchem_df = uncleaned_wetchem_df.copy(deep=True)
    wetchem_df.set_index("sample_code")

    quartiles_dict = {}
    for chem in undefined_chems:
        quartiles_dict[chem] = {}
        upper_quartile = wetchem_df[chem].quantile(0.75)
        median = wetchem_df[chem].quantile(0.50)
        lower_quartile = wetchem_df[chem].quantile(0.25)
        quartiles_dict[chem]['Value_1'] = lower_quartile
        quartiles_dict[chem]['Value_2'] = median
        quartiles_dict[chem]['Value_3'] = upper_quartile

    pcc_dict = {**pcc_classes_dict, **quartiles_dict}
    pd.DataFrame(pcc_dict).T.to_csv('outputFiles/pcc_sumnmnary_statistics.csv')


    return redbooth_outliers, pcc_dict

# load_residual_outliers()