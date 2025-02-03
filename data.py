import psycopg2
import pandas as pd
import ast
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path
import glob
import pyodbc

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
# get_spc()

def getWetchem(chemicals=["organic_carbon"]):
    sample_codes = pd.read_csv("outputFiles/spectra.csv",index_col=0).index
    cn_chems = ['organic_carbon', 'total_nitrogen']
    cn = pd.DataFrame()
    for file in Path("/home/tom/DSML124/inputFiles/CN_files").rglob("**/*.xlsx"):
        excel_file = pd.ExcelFile(str(file))
        sheet_names = excel_file.sheet_names
        for sheet_name in sheet_names:
            print(f"Reading sheet: {sheet_name}")
            
            cn_df = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=5)
            for col in cn_df.columns:
                cn_df = cn_df.rename(columns={col: str(col).strip().replace(" ","_").lower()})
                print(cn_df.columns)
                cn_df = cn_df.rename(columns={"c":"organic_carbon", "n":"total_nitrogen"}) 
                print(cn_df.columns)
                columns = ["sample_code","total_nitrogen","organic_carbon"]
                for col in cn_df.columns:
                    if col not in columns:
                        cn_df = cn_df.drop(col, axis=1)
            if ("sample_code" in cn_df.columns) and ("organic_carbon" in cn_df.columns) and ("total_nitrogen" in cn_df.columns):
                cn_df['sample_code'] = [str(i).split(" ")[0] for i in cn_df['sample_code']]
                cn = pd.concat([cn,cn_df])

    if "sample_code" not in cn.columns:
        cn = cn.rename(columns={"Unnamed: 0":"sample_code"})
    for file in Path("/home/tom/DSML125/inputFiles/CN_files").rglob("**/*.csv"):
        _ = pd.read_csv(file)
        _ = _.reset_index()
        cn= pd.concat([cn, _])

    conn_lims = pyodbc.connect("Driver={/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.4.so.1.1};"
                            "TrustServerCertificate=yes;"
                            "Server=192.168.5.18\CROPNUT;"
                            "Database=cropnuts;"
                            "uid=thomasTsuma;pwd=GR^KX$uRe9#JwLc6")
    
    wet = pd.DataFrame(columns=["sample_code"])
    
    if(len(sample_codes) < 5000):
        count  = len(sample_codes)
        step=count
    elif(len(sample_codes) < 200000):
        count = len(sample_codes)
        step=5000
    else:
        count = len(sample_codes)
        step=5000
    start = 0

    # start = 0
    # step = 500
    # count = 2000
    
    # not_null_clause = ""
    chemicals_clause = ""
    for c in [ i for i in chemicals if i not in cn_chems]:
        if( c == 'cec'):
            chemicals_clause = chemicals_clause + f"LOWER(chemical_name) LIKE '%c.e.c%' OR "
        elif( c == 'ec_salts'):
            chemicals_clause = chemicals_clause + f"LOWER(chemical_name) LIKE '%ec_(salts)%' OR "
        elif( c == 'psi'):
            chemicals_clause = chemicals_clause + f"LOWER(chemical_name) LIKE '%Phosphorus Sorption Index (PSI)%' OR "   
        elif( c == 'reactive_carbon'):
            chemicals_clause = chemicals_clause + f"LOWER(chemical_name) LIKE '%Reactive Carbon%' OR "    
        elif( c == 'phosphorus_olsen'):
            chemicals_clause = chemicals_clause + f"LOWER(chemical_name) LIKE '%Phosphorus(Olsen)%' OR "                   
        else:
            chemicals_clause = chemicals_clause + f"LOWER(chemical_name) LIKE '{c}' OR "
    # not_null_clause = " ".join(not_null_clause.split(" ")[:-2])
    chemicals_clause = " ".join(chemicals_clause.split(" ")[:-2])
    print(chemicals_clause)
    wet_ = pd.DataFrame()
    if chemicals_clause.strip() != "":
        for i in np.arange(start, count, step):
            print("Fetching wetchem from {}".format(start))
            samples = [i for i in sample_codes][start:start+step]
            _ = pd.read_sql_query(f"SELECT sample_code, result, chemical_name from SampleResults where ({chemicals_clause}) AND sample_code IN {str(samples).replace('[','(').replace(']',')')} ORDER BY processed_date ASC", con=conn_lims)
            wet_ = pd.concat([wet_, _], axis=0)
            start = start + step
            if (count-step) > 5000:
                step=5000
            else:
                step = count-step
    
    
        wet_ = wet_.replace('Phosphorus Sorption Index (PSI)','psi')
        wet_ = wet_.replace('Reactive Carbon','reactive_carbon')
        wet_ = wet_.replace('Phosphorus(Olsen)','phosphorus_olsen')
        
        wet = pd.pivot_table(index='sample_code', values="result", columns="chemical_name", data=wet_, aggfunc=max)
        wet = wet.reset_index()
        wet.columns = [i.lower().replace(" ","_").replace(".","").replace("(","").replace(")","").strip() for i in wet.columns]
        wet['sample_code'] = wet.sample_code.str.strip()
    wet.columns = [i.lower().replace(" ","_").replace(".","").replace("(","").replace(")","").strip() for i in wet.columns]
    wet['sample_code'] = wet.sample_code.str.strip()
    print(wet.columns)
    # for c in chemicals:
    #     if(c not in wet.columns):
    #         raise Exception(f"{c} not in wetchem data")
    if "sample_code" not in cn.columns:
        cn = cn.rename(columns={'Unnamed: 0':"sample_code"})

    uncleaned_wetchem_df = wet.copy()
    if len(cn) > 0:
        uncleaned_wetchem_df = pd.merge(uncleaned_wetchem_df, cn, on="sample_code", how="outer")

    print(cn.columns)
    # uncleaned_wetchem_df = uncleaned_wetchem_df.rename(columns={"Unnamed: 0": "sample_code"})
    uncleaned_wetchem_df.set_index("sample_code")
    for column in uncleaned_wetchem_df.columns:
        if (column != 'sample_code'):
            vals = []
            for value in uncleaned_wetchem_df[column].values:
                if (value is not None):
                    value = str(value)
                    value = value.replace(">", "").replace(
                        "<", "").replace("...", "").strip()
                    try:
                        value = float(value)
                    except:
                        value = np.nan
                vals.append(value)
            uncleaned_wetchem_df[column] = vals

    wetchem_df = uncleaned_wetchem_df.copy(deep=True)
    wetchem_df.to_csv("outputFiles/cleaned_wetchem.csv")

    return wetchem_df

getWetchem()

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