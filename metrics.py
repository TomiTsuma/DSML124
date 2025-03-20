import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import sys
sys.path.append('/home/tom/DSML124/QC_Model_Predictions')
from predict import predict_chems
from pathlib import Path
import pyodbc
import numpy as np
import os
from data import load_residual_outliers
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report




# Define API scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Authenticate using credentials.json
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Open the Google Sheet
spreadsheet = client.open("Spectral Outliers (2025-03-20 07-17-05)")  # Change to your Google Sheet name
worksheet = spreadsheet.worksheet("Sheet1")  # Change to your sheet name

# Read data into Pandas DataFrame
data = worksheet.get_all_records()
df = pd.DataFrame(data)
df.to_csv("test/outlier_check.csv")

print(df.head()) 

chemicals = ["ph", "phosphorus", "potassium", "calcium",  "boron", "copper", "zinc", "clay", "silt", "sand" , "total_nitrogen", "organic_carbon" , "exchangeable_acidity"]
#
spc = pd.read_csv("test/spc.csv",engine='c',index_col=0)
print(spc.loc[spc.index.isin(['CM746SA0093'])])

# predict_chems(
#     '/home/tom/DSML124/QC_Model_Predictions/dl_models_all_chems_20210414/v2.3',
#     '/home/tom/DSML124/test/predictions',
#     chemicals,
#     ['v2.3'],
#     spc
#     )

def getWetchem(chemicals=["organic_carbon"], sample_codes=[]):
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
    for file in Path("/home/tom/DSML124/inputFiles/CN_files").rglob("**/*.csv"):
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
    wetchem_df.to_csv("test/cleaned_wetchem.csv")


# getWetchem(chemicals=chemicals, sample_codes=df['sample_code'].tolist())



def residual_outliers(chems, model_version):
    print("Getting residual outliers for preds vs wetchem")
    spectra = pd.read_csv('test/spc.csv', index_col=0, engine='c')
    wetchem_df = pd.read_csv("test/cleaned_wetchem.csv")
    print(len(wetchem_df))

    redbooth_outliers_dict, pcc_classes_dict = load_residual_outliers(chemicals=chems)

    os.makedirs('test/PCC1', exist_ok=True)
    os.makedirs('test/PCC2', exist_ok=True)
    os.makedirs('test/PCC3', exist_ok=True)
    os.makedirs('test/PCC_Classes', exist_ok=True)
    for chem in chems:
        wet = wetchem_df.loc[wetchem_df[chem].notnull()]
        df = pd.read_csv(f"./test/predictions/{model_version}/{chem}_preds.csv")
        df = df.rename(columns={'Unnamed: 0':'sample_code'})
        if('sample_code' not in df.columns):
            df = df.rename(columns={'sample_id':'sample_code'})
        df = pd.merge(df, wetchem_df, on='sample_code', how="inner")
        df = df.loc[df[chem].notnull()]
        df = df[['sample_code', chem, '0']]
        

        df['Difference'] =  df['0'] - df[chem]

        

        df.to_csv(f"test/PCC_Classes/{chem}.csv")
        if chem == 'organic_carbon':
            df['residual_outlier_limit'] = df[chem].apply(lambda x: 0.2 * x if x > 1.5 else None)
            df.loc[df[chem] < 1.5, 'residual_outlier_limit'] = 0.3
            df['Difference'] = abs(df['Difference'])
            df.loc[df['Difference'] <= df['residual_outlier_limit'],'PCC_Class'] = 1
            df.loc[df['Difference'] > df['residual_outlier_limit'],'PCC_Class'] = 3
            df.to_csv(f"/home/tom/DSML124/test/PCC_Classes/{chem}.csv")
            spectra.loc[spectra.index.isin(df.loc[df['Difference'] <= df['residual_outlier_limit']].sample_code)].to_csv(f'/home/tom/DSML124/test/PCC1/{chem}.csv')
            spectra.loc[spectra.index.isin(df.loc[df['Difference'] > df['residual_outlier_limit']].sample_code)].to_csv(f'/home/tom/DSML124/test/PCC3/{chem}.csv')
        elif chem == 'total_nitrogen':
            df['residual_outlier_limit'] = df[chem].apply(lambda x: 0.2 * x if x > 0.15 else None)
            df.loc[df[chem] < 0.15, 'residual_outlier_limit'] = 0.03
            df['Difference'] = abs(df['Difference'])
            df.loc[df['Difference'] <= df['residual_outlier_limit'],'PCC_Class'] = 1
            df.loc[df['Difference'] > df['residual_outlier_limit'],'PCC_Class'] = 3
            df.to_csv(f"/home/tom/DSML124/test/PCC_Classes/{chem}.csv")
            spectra.loc[spectra.index.isin(df.loc[df['Difference'] > df['residual_outlier_limit']].sample_code)].to_csv(f'/home/tom/DSML124/test/PCC3/{chem}.csv')
            spectra.loc[spectra.index.isin(df.loc[df['Difference'] <= df['residual_outlier_limit']].sample_code)].to_csv(f'/home/tom/DSML124/test/PCC1/{chem}.csv')
        elif (chem in redbooth_outliers_dict.keys()):
            print(chem)
            lower = redbooth_outliers_dict[chem][0]
            upper = redbooth_outliers_dict[chem][1]
            df.loc[(df['Difference'] < lower) | (df['Difference'] > upper),'PCC_Class'] = 3
            df.loc[(df['Difference'] > lower) & (df['Difference'] < upper),'PCC_Class'] = 1
            df.to_csv(f"/home/tom/DSML124/test/PCC_Classes/{chem}.csv")
            spectra.loc[spectra.index.isin(df.loc[(df['Difference']> lower) & (df['Difference'] < upper)]['sample_code'])].to_csv(f'test/PCC1/{chem}.csv')
            spectra.loc[spectra.index.isin(df.loc[(df['Difference']< lower) | (df['Difference'] > upper)]['sample_code'])].to_csv(f'test/PCC3/{chem}.csv')
        elif(chem in pcc_classes_dict.keys()):
            print(chem)
            lower = None
            mid = None
            upper = None
            lower = pcc_classes_dict[chem]['Value_1']
            mid = pcc_classes_dict[chem]['Value_2']
            upper = pcc_classes_dict[chem]['Value_3']
            if(chem in df.columns):
                df.loc[df[chem] < lower, 'Actual_PCC'] = 1
                df.loc[df['0'] < lower, 'Predicted_PCC'] = 1

                df.loc[(df[chem] >= lower) & (df[chem] < mid), 'Actual_PCC'] = 2
                df.loc[(df['0'] >= lower) & (df[chem] < mid), 'Predicted_PCC'] = 2

                df.loc[(df[chem] >= mid) & (df[chem] < upper), 'Actual_PCC'] = 3
                df.loc[(df['0'] >= mid) & (df[chem] < upper), 'Predicted_PCC'] = 3

                df.loc[df[chem] >= upper, 'Actual_PCC'] = 4
                df.loc[df['0'] >= upper, 'Predicted_PCC'] = 4

                df['PCC_Class'] = (df['Actual_PCC'] - df['Predicted_PCC']).abs()

                df.to_csv(f"test/PCC_Classes/{chem}.csv")

                spectra.loc[spectra.index.str.strip().isin(df.loc[df['PCC_Class'] <= 1]['sample_code'].values)].to_csv(f'test/PCC1/{chem}.csv')
                spectra.loc[spectra.index.str.strip().isin(df.loc[df['PCC_Class'] == 2]['sample_code'].values)].to_csv(f'test/PCC2/{chem}.csv')
                spectra.loc[spectra.index.str.strip().isin(df.loc[df['PCC_Class'] >= 3]['sample_code'].values)].to_csv(f'test/PCC3/{chem}.csv')            
        else:
            continue


# residual_outliers(chemicals, "v2.3")


for chemical in chemicals:

    print(chemical)
    df_ = df.copy()
    pcc_classes = pd.read_csv(f"test/PCC_Classes/{chemical}.csv")
    pcc_classes.loc[pcc_classes['PCC_Class'] > 1, 'is_outlier'] = True
    pcc_classes.loc[pcc_classes['PCC_Class'] <= 1, 'is_outlier'] = False

    print(len(pcc_classes.loc[pcc_classes['PCC_Class'] <= 1]))
    print(len(pcc_classes.loc[pcc_classes['PCC_Class'] > 1]))
    continue
    
    pcc_classes['is_outlier'] = pcc_classes['is_outlier'].astype(bool)
    df_ = df_.replace('TRUE', "Outlier")
    df_ = df_.replace('FALSE', "Not Outlier")
    print(df)
    pcc_classes = pcc_classes.replace(True, "Outlier")
    pcc_classes = pcc_classes.replace(False, "Not Outlier")
    # df_=df_.dropna()
    
    df_ = df_.dropna(subset=chemical)

    _ = pd.merge(df_[[chemical,'sample_code']],pcc_classes[['sample_code','is_outlier']], on="sample_code", how="inner")
    clf_report = classification_report(_['is_outlier'], _[chemical], output_dict=True)
    print(clf_report)
    pd.DataFrame(clf_report).transpose().to_csv(f"./test/clf_reports/{chemical}.csv")
    cm = confusion_matrix(_['is_outlier'], _[chemical])
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Outlier", "Not Outlier"], yticklabels=["Outlier", "Not Outlier"])

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"./test/visualizations/confusion_matrix_{chemical}.png", dpi=300, bbox_inches="tight")
    # break


