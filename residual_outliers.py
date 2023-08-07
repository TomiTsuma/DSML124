import pandas as pd
import os
import numpy as np
from data import load_residual_outliers
# import rpy2.robjects as robjects
# r = robjects.r
# r['source']('pcc_V1.7.r')
# pcc = robjects.globalenv['pcc']
# from rpy2.robjects import pandas2ri, numpy2ri
# pandas2ri.activate()
# numpy2ri.activate()

# data = pd.read_csv('outputFiles/PCC_Classes/boron.csv', index_col=0)
# pcc(
#     "boron",
#     numpy2ri.py2rpy(np.array([0.5,0.8,1])),
#     pandas2ri.py2rpy(data),
#     1,
#     2
#     )



# uncleaned_wetchem_df = pd.read_csv("inputFiles/all_wetchem_data_uncleeaned_15-7-2022 (1).csv")
# uncleaned_wetchem_df = uncleaned_wetchem_df.rename(columns={"Unnamed: 0":"sample_code"})
# uncleaned_wetchem_df.set_index("sample_code")
# for column in uncleaned_wetchem_df.columns:
#     if(column != 'sample_code'):
#         vals = []
#         for value in uncleaned_wetchem_df[column].values:
#             if(value is not None):
#                 value = str(value)
#                 value = value.replace(">","").replace("<","").replace("...","").strip()
#                 value = float(value)
#             vals.append(value)
#         uncleaned_wetchem_df[column] = vals

# wetchem_df = uncleaned_wetchem_df.copy(deep=True)
# wetchem_df.set_index("sample_code")

# wetchem_df.to_csv("inputFiles/cleaned_wetchem.csv")

def residual_outliers():
    print("Getting residual outliers")
    spectra = pd.read_csv('outputFiles/spectra.csv', index_col=0, engine='c')
    wetchem_df = pd.read_csv("inputFiles/cleaned_wetchem.csv")

    redbooth_outliers_dict, pcc_classes_dict = load_residual_outliers()


    os.makedirs('outputFiles/PCC1', exist_ok=True)
    os.makedirs('outputFiles/PCC2', exist_ok=True)
    os.makedirs('outputFiles/PCC3', exist_ok=True)
    os.makedirs('outputFiles/PCC_Classes', exist_ok=True)
    for file in os.listdir("outputFiles/predictions/DLv2.2"):
        chem = file.split('.')[0]
        wet = wetchem_df.loc[wetchem_df[chem].notnull()]
        df = pd.read_csv(f"./outputFiles/predictions/DLv2.2/{file}")
        df = df.rename(columns={'Unnamed: 0':'sample_code'})
        if('sample_code' not in df.columns):
            df = df.rename(columns={'sample_id':'sample_code'})
        df = pd.merge(df, wetchem_df, on='sample_code', how="inner")
        df = df.loc[df[chem].notnull()]
        df = df[['sample_code', chem, '0']]
        df['Difference'] =  df['0'] - df[chem]

        df.to_csv(f"outputFiles/PCC_Classes/{chem}.csv")
        if(chem in redbooth_outliers_dict.keys()):
            lower = redbooth_outliers_dict[chem][0]
            upper = redbooth_outliers_dict[chem][1]
            spectra.loc[spectra.index.isin(df.loc[(df['Difference']> lower) | (df['Difference'] < upper)]['sample_code'])].to_csv(f'outputFiles/PCC1/{chem}.csv')
            spectra.loc[spectra.index.isin(df.loc[(df['Difference']< lower) | (df['Difference'] > upper)]['sample_code'])].to_csv(f'outputFiles/PCC3/{chem}.csv')
        elif(chem in pcc_classes_dict.keys()):
            lower = None
            mid = None
            upper = None
            print(chem)
            lower = pcc_classes_dict[chem]['Value_1']
            mid = pcc_classes_dict[chem]['Value_2']
            upper = pcc_classes_dict[chem]['Value_3']
            if(chem in df.columns):
                df.loc[df[chem] < lower, 'Actual_PCC'] = 1
                df.loc[df['0'] < lower, 'Predicted_PCC'] = 1

                df.loc[(df[chem] > lower) & (df[chem] < mid), 'Actual_PCC'] = 2
                df.loc[(df['0'] > lower) & (df[chem] < mid), 'Predicted_PCC'] = 2

                df.loc[(df[chem] > mid) & (df[chem] < upper), 'Actual_PCC'] = 3
                df.loc[(df['0'] > mid) & (df[chem] < upper), 'Predicted_PCC'] = 3

                df.loc[df[chem] > upper, 'Actual_PCC'] = 4
                df.loc[df['0'] > upper, 'Predicted_PCC'] = 4

                df['PCC_Class'] = (df['Actual_PCC'] - df['Predicted_PCC']).abs()

                df.to_csv(f"outputFiles/PCC_Classes/{chem}.csv")
                spectra.loc[spectra.index.isin(df.loc[df['PCC_Class'] <= 1]['sample_code'])].to_csv(f'outputFiles/PCC1/{chem}.csv')
                spectra.loc[spectra.index.isin(df.loc[df['PCC_Class'] == 2]['sample_code'])].to_csv(f'outputFiles/PCC2/{chem}.csv')
                spectra.loc[spectra.index.isin(df.loc[df['PCC_Class'] >= 3]['sample_code'])].to_csv(f'outputFiles/PCC3/{chem}.csv')            
        else:
            continue


def residual_outliers_reconstructed():
    print("Getting residual outliers")
    redbooth_outliers_dict, pcc_classes_dict = load_residual_outliers()


    os.makedirs('outputFiles/PCC1_Classes_Reconstructed', exist_ok=True)
    os.makedirs('outputFiles/PCC3_Classes_Reconstructed', exist_ok=True)

    for file in os.listdir("outputFiles/predictions/pcc1_autoencoded/DLv2.2"):
        print(file)
        chem = file.split('.')[0]
        dl_predictions = pd.read_csv(f"outputFiles/predictions/DLv2.2/{file}")
        dl_predictions = dl_predictions.rename(columns={'0':chem})
        print(dl_predictions.head(5))
        autoencoded_predictions = pd.read_csv(f"outputFiles/predictions/pcc1_autoencoded/DLv2.2/{file}")
        print(autoencoded_predictions.head(5))
        df = pd.merge(dl_predictions, autoencoded_predictions, on="Unnamed: 0", how="inner")
        print(df.head(5))
        if('sample_code' not in df.columns):
            df = df.rename(columns={"Unnamed: 0":'sample_code'})
        # break
        df = df.loc[df[chem].notnull()]
        df = df[['sample_code', chem, '0']]
        df['Difference'] =  (df['0'] - df[chem]).abs()

        # df.to_csv(f"outputFiles/PCC_Classes/{chem}.csv")
        if(chem in redbooth_outliers_dict.keys()):
            lower = redbooth_outliers_dict[chem][0]
            upper = redbooth_outliers_dict[chem][1]
            df.loc[(df['Difference']> lower) & (df['Difference'] < upper), 'PCC_Class'] = 1
            df.loc[(df['Difference']< lower) | (df['Difference'] > upper), 'PCC_Class'] = 3
            df.to_csv(f'outputFiles/PCC1_Classes_Reconstructed/{chem}.csv')
        elif(chem in pcc_classes_dict.keys()):
            lower = None
            mid = None
            upper = None
            print(chem)
            lower = pcc_classes_dict[chem]['Value_1']
            mid = pcc_classes_dict[chem]['Value_2']
            upper = pcc_classes_dict[chem]['Value_3']
            if(chem in df.columns):
                df.loc[df[chem] <= lower, 'Actual_PCC'] = 1
                df.loc[df['0'] <= lower, 'Predicted_PCC'] = 1

                df.loc[(df[chem] > lower) & (df[chem] <= mid), 'Actual_PCC'] = 2
                df.loc[(df['0'] > lower) & (df['0'] <= mid), 'Predicted_PCC'] = 2

                df.loc[(df[chem] > mid) & (df[chem] <= upper), 'Actual_PCC'] = 3
                df.loc[(df['0'] > mid) & (df['0'] <= upper), 'Predicted_PCC'] = 3

                df.loc[df[chem] > upper, 'Actual_PCC'] = 4
                df.loc[df['0'] > upper, 'Predicted_PCC'] = 4

                df['PCC_Class'] = (df['Actual_PCC'] - df['Predicted_PCC']).abs()

                df.to_csv(f"outputFiles/PCC1_Classes_Reconstructed/{chem}.csv")
           
        else:
            continue

    for file in os.listdir("outputFiles/predictions/pcc3_autoencoded/DLv2.2"):
        print(file)
        chem = file.split('.')[0]
        dl_predictions = pd.read_csv(f"outputFiles/predictions/DLv2.2/{file}")
        dl_predictions = dl_predictions.rename(columns={'0':chem})
        print(dl_predictions.head(5))
        autoencoded_predictions = pd.read_csv(f"outputFiles/predictions/pcc3_autoencoded/DLv2.2/{file}")
        print(autoencoded_predictions.head(5))
        df = pd.merge(dl_predictions, autoencoded_predictions, on="Unnamed: 0", how="inner")
        print(df.head(5))
        if('sample_code' not in df.columns):
            df = df.rename(columns={"Unnamed: 0":'sample_code'})
        # break
        df = df.loc[df[chem].notnull()]
        df = df[['sample_code', chem, '0']]
        df['Difference'] =  (df['0'] - df[chem]).abs()

        # df.to_csv(f"outputFiles/PCC_Classes/{chem}.csv")
        if(chem in redbooth_outliers_dict.keys()):
            lower = redbooth_outliers_dict[chem][0]
            upper = redbooth_outliers_dict[chem][1]
            df.loc[(df['Difference']> lower) & (df['Difference'] < upper), 'PCC_Class'] = 1
            df.loc[(df['Difference']< lower) | (df['Difference'] > upper), 'PCC_Class'] = 3
            df.to_csv(f'outputFiles/PCC3_Classes_Reconstructed/{chem}.csv')
        elif(chem in pcc_classes_dict.keys()):
            lower = None
            mid = None
            upper = None
            print(chem)
            lower = pcc_classes_dict[chem]['Value_1']
            mid = pcc_classes_dict[chem]['Value_2']
            upper = pcc_classes_dict[chem]['Value_3']
            if(chem in df.columns):
                df.loc[df[chem] <= lower, 'Actual_PCC'] = 1
                df.loc[df['0'] <= lower, 'Predicted_PCC'] = 1

                df.loc[(df[chem] > lower) & (df[chem] <= mid), 'Actual_PCC'] = 2
                df.loc[(df['0'] > lower) & (df['0'] <= mid), 'Predicted_PCC'] = 2

                df.loc[(df[chem] > mid) & (df[chem] <= upper), 'Actual_PCC'] = 3
                df.loc[(df['0'] > mid) & (df['0'] <= upper), 'Predicted_PCC'] = 3

                df.loc[df[chem] > upper, 'Actual_PCC'] = 4
                df.loc[df['0'] > upper, 'Predicted_PCC'] = 4

                df['PCC_Class'] = (df['Actual_PCC'] - df['Predicted_PCC']).abs()

                df.to_csv(f"outputFiles/PCC3_Classes_Reconstructed/{chem}.csv")
           
        else:
            continue


# residual_outliers_reconstructed()