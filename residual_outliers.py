import pandas as pd
import os
import numpy as np
from data import load_residual_outliers



def residual_outliers(chems):
    print("Getting residual outliers for preds vs wetchem")
    spectra = pd.read_csv('outputFiles/v1_spectra.csv', index_col=0, engine='c')
    wetchem_df = pd.read_csv("inputFiles/cleaned_wetchem.csv")
    print(len(wetchem_df))

    redbooth_outliers_dict, pcc_classes_dict = load_residual_outliers()

    os.makedirs('outputFiles/PCC1', exist_ok=True)
    os.makedirs('outputFiles/PCC2', exist_ok=True)
    os.makedirs('outputFiles/PCC3', exist_ok=True)
    os.makedirs('outputFiles/PCC_Classes', exist_ok=True)
    for chem in chems:
        wet = wetchem_df.loc[wetchem_df[chem].notnull()]
        df = pd.read_csv(f"./outputFiles/predictions/DLv2.2/{chem}.csv")
        df = df.rename(columns={'Unnamed: 0':'sample_code'})
        if('sample_code' not in df.columns):
            df = df.rename(columns={'sample_id':'sample_code'})
        df = pd.merge(df, wetchem_df, on='sample_code', how="inner")
        df = df.loc[df[chem].notnull()]
        df = df[['sample_code', chem, '0']]
        

        df['Difference'] =  df['0'] - df[chem]

        

        df.to_csv(f"outputFiles/PCC_Classes/{chem}.csv")
        if(chem in redbooth_outliers_dict.keys()):
            print(chem)
            lower = redbooth_outliers_dict[chem][0]
            upper = redbooth_outliers_dict[chem][1]
            spectra.loc[spectra.index.isin(df.loc[(df['Difference']> lower) | (df['Difference'] < upper)]['sample_code'])].to_csv(f'outputFiles/PCC1/{chem}.csv')
            spectra.loc[spectra.index.isin(df.loc[(df['Difference']< lower) | (df['Difference'] > upper)]['sample_code'])].to_csv(f'outputFiles/PCC3/{chem}.csv')
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

                df.to_csv(f"outputFiles/PCC_Classes/{chem}.csv")

                spectra.loc[spectra.index.str.strip().isin(df.loc[df['PCC_Class'] <= 1]['sample_code'].values)].to_csv(f'outputFiles/PCC1/{chem}.csv')
                spectra.loc[spectra.index.str.strip().isin(df.loc[df['PCC_Class'] == 2]['sample_code'].values)].to_csv(f'outputFiles/PCC2/{chem}.csv')
                spectra.loc[spectra.index.str.strip().isin(df.loc[df['PCC_Class'] >= 3]['sample_code'].values)].to_csv(f'outputFiles/PCC3/{chem}.csv')            
        else:
            continue

def residual_outliers_reconstructed(chems):
    print("Getting residual outliers reconstructed")
    redbooth_outliers_dict, pcc_classes_dict = load_residual_outliers()


    os.makedirs('outputFiles/PCC1_Classes_Reconstructed', exist_ok=True)
    os.makedirs('outputFiles/PCC2_Classes_Reconstructed', exist_ok=True)

    for chem in chems:
        dl_predictions = pd.read_csv(f"outputFiles/predictions/DLv2.2/{chem}.csv")
        dl_predictions = dl_predictions.rename(columns={'0':chem})
        
        autoencoded_predictions = pd.read_csv(f"outputFiles/predictions/pcc1_autoencoded/DLv2.2/{chem}.csv")
        
        if("sample_code" not in autoencoded_predictions.columns):
            autoencoded_predictions = autoencoded_predictions.rename(columns={'Unnamed: 0':'sample_code'})
        if("sample_code" not in dl_predictions.columns):
            dl_predictions = dl_predictions.rename(columns={'Unnamed: 0':'sample_code'})
        df = pd.merge(dl_predictions, autoencoded_predictions, on="sample_code", how="inner")
        print(chem)
        assert len(df) > 0
        # print(df.head(5))
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

    for chem in chems:
        dl_predictions = pd.read_csv(f"outputFiles/predictions/DLv2.2/{chem}.csv")
        dl_predictions = dl_predictions.rename(columns={'0':chem})
        autoencoded_predictions = pd.read_csv(f"outputFiles/predictions/pcc2_autoencoded/DLv2.2/{chem}.csv")
        if('sample_code' not in dl_predictions.columns):
            dl_predictions = dl_predictions.rename(columns={"Unnamed: 0":'sample_code'})
        if('sample_code' not in autoencoded_predictions.columns):
            autoencoded_predictions = autoencoded_predictions.rename(columns={"Unnamed: 0":'sample_code'})
        df = pd.merge(dl_predictions, autoencoded_predictions, on="sample_code", how="inner")
        
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
            df.to_csv(f'outputFiles/PCC2_Classes_Reconstructed/{chem}.csv')
        elif(chem in pcc_classes_dict.keys()):
            lower = None
            mid = None
            upper = None
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

                df.to_csv(f"outputFiles/PCC2_Classes_Reconstructed/{chem}.csv")
           
        else:
            continue


def residual_outliers_reconstructed_wetchem(chems):
    print("Getting residual outliers")
    os.makedirs("outputFiles/PCC1_Wetchem_Reconstructed", exist_ok=True)
    os.makedirs("outputFiles/PCC2_Wetchem_Reconstructed", exist_ok=True)
    wetchem_df = pd.read_csv("inputFiles/cleaned_wetchem.csv")

    redbooth_outliers_dict, pcc_classes_dict = load_residual_outliers()

    for chem in chems:
        wet = wetchem_df.loc[wetchem_df[chem].notnull()]
        df = pd.read_csv(f"./outputFiles/predictions/pcc1_autoencoded/DLv2.2/{chem}.csv")
        df = df.rename(columns={'Unnamed: 0':'sample_code'})
        if('sample_code' not in df.columns):
            df = df.rename(columns={'Unnamed: 0':'sample_code'})
        df = pd.merge(df, wetchem_df, on='sample_code', how="inner")
        df = df.loc[df[chem].notnull()]
        df = df[['sample_code', chem, '0']]
        df['Difference'] =  df['0'] - df[chem]

        assert len(df) > 0
        
        df.to_csv(f"outputFiles/PCC1_Wetchem_Reconstructed/{chem}.csv")
        if(chem in redbooth_outliers_dict.keys()):
            lower = redbooth_outliers_dict[chem][0]
            upper = redbooth_outliers_dict[chem][1]
        elif(chem in pcc_classes_dict.keys()):
            lower = None
            mid = None
            upper = None
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

                df.to_csv(f"outputFiles/PCC1_Wetchem_Reconstructed/{chem}.csv")               
        else:
            continue

    for chem in chems:
        wet = wetchem_df.loc[wetchem_df[chem].notnull()]
        df = pd.read_csv(f"./outputFiles/predictions/pcc2_autoencoded/DLv2.2/{chem}.csv")
        df = df.rename(columns={'Unnamed: 0':'sample_code'})
        if('sample_code' not in df.columns):
            df = df.rename(columns={'sample_id':'sample_code'})
        df = pd.merge(df, wetchem_df, on='sample_code', how="inner")
        df = df.loc[df[chem].notnull()]
        df = df[['sample_code', chem, '0']]
        df['Difference'] =  df['0'] - df[chem]

        df.to_csv(f"outputFiles/PCC3_Wetchem_Reconstructed/{chem}.csv")
        if(chem in redbooth_outliers_dict.keys()):
            lower = redbooth_outliers_dict[chem][0]
            upper = redbooth_outliers_dict[chem][1]
        elif(chem in pcc_classes_dict.keys()):
            lower = None
            mid = None
            upper = None
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

                df.to_csv(f"outputFiles/PCC3_Wetchem_Reconstructed/{chem}.csv")               
        else:
            continue




