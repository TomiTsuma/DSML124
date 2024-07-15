import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

def pcc2_confusion_matrix(chemical):
    orig_PCC2 = pd.read_csv(f"outputFiles/PCC2/{chemical}.csv")
    reconstructed_class = pd.read_csv(f"outputFiles/PCC2_Classes_Reconstructed/{chemical}.csv")
    orig_PCC2 = orig_PCC2[['Unnamed: 0']]
    orig_PCC2['PCC_Class'] = 2
    reconstructed_class.loc[reconstructed_class['PCC_Class'] > 0, 'PCC_Class'] =2

    reconstructed_class = reconstructed_class.set_index("Unnamed: 0")
    orig_PCC2 = orig_PCC2.set_index("Unnamed: 0")

    reconstructed_class = reconstructed_class.reindex(orig_PCC2.index)

    print(reconstructed_class)
    print(orig_PCC2)
    os.makedirs(f"outputFiles/visualizations/{chemical}", exist_ok=True)
    target_names =[i for i in np.unique(reconstructed_class['PCC_Class'].values)]
    fig = ff.create_annotated_heatmap(z=confusion_matrix(orig_PCC2['PCC_Class'].values, reconstructed_class['PCC_Class'].values),x=target_names, y=target_names, colorscale='blues', showscale=False, reversescale=False)
    fig['layout']['xaxis'].update(side='bottom', title='Predicted')
    fig['layout']['yaxis'].update(side='left', title='True')
    fig.update_layout(title=f'confusion matrix', width=1000, height=1000)
    fig.write_image(f"outputFiles/visualizations/{chemical}/PCC2_confusion_matrix.png")




def pcc3_confusion_matrix(chemical):
    orig_PCC3 = pd.read_csv(f"outputFiles/PCC3/{chemical}.csv")
    reconstructed_class = pd.read_csv(f"outputFiles/PCC3_Classes_Reconstructed/{chemical}.csv")
    
    orig_PCC3 = orig_PCC3[['Unnamed: 0']]
    orig_PCC3['PCC_Class'] = 3
    reconstructed_class.loc[reconstructed_class['PCC_Class'] > 0, 'PCC_Class'] =3
    
    reconstructed_class = reconstructed_class.set_index("Unnamed: 0")
    orig_PCC3 = orig_PCC3.set_index("Unnamed: 0")

    # reconstructed_class = reconstructed_class.reindex(orig_PCC3.index)

    print(reconstructed_class)
    print(orig_PCC3)
    os.makedirs(f"outputFiles/visualizations/{chemical}", exist_ok=True)
    target_names =[i for i in np.unique(reconstructed_class['PCC_Class'].values)]
    fig = ff.create_annotated_heatmap(z=confusion_matrix(orig_PCC3['PCC_Class'].values, reconstructed_class['PCC_Class'].values),x=target_names, y=target_names, colorscale='blues', showscale=False, reversescale=False)
    fig['layout']['xaxis'].update(side='bottom', title='Predicted')
    fig['layout']['yaxis'].update(side='left', title='True')
    fig.update_layout(title=f'confusion matrix', width=1000, height=1000)
    fig.write_image(f"outputFiles/visualizations/{chemical}/PCC3_confusion_matrix.png")

def pcc1_confusion_matrix(chemical):
    orig_PCC1 = pd.read_csv(f"outputFiles/PCC1/{chemical}.csv")
    reconstructed_class = pd.read_csv(f"outputFiles/PCC1_Classes_Reconstructed/{chemical}.csv")
    orig_PCC1 = orig_PCC1[['Unnamed: 0']]
    orig_PCC1['PCC_Class'] = 1
    reconstructed_class['PCC_Class'] = reconstructed_class['PCC_Class'].astype('int')
    reconstructed_class.loc[reconstructed_class['PCC_Class'] > 0, 'PCC_Class'] = 3
    
    reconstructed_class = reconstructed_class.set_index("sample_code")
    orig_PCC1 = orig_PCC1.set_index("Unnamed: 0")

    orig_PCC1 = orig_PCC1.reindex(reconstructed_class.index)

    print(reconstructed_class['PCC_Class'])
    print(orig_PCC1['PCC_Class'])
    os.makedirs(f"outputFiles/visualizations/{chemical}", exist_ok=True)
    target_names =[i for i in np.unique(orig_PCC1['PCC_Class'].values)]
    # target_names = [1,2,3]
    print(target_names)
    fig = ff.create_annotated_heatmap(z=confusion_matrix(orig_PCC1['PCC_Class'].values, reconstructed_class['PCC_Class'].values),x=target_names, y=target_names, colorscale='blues', showscale=False, reversescale=False)
    fig['layout']['xaxis'].update(side='bottom', title='Predicted')
    fig['layout']['yaxis'].update(side='left', title='True')
    fig.update_layout(title=f'confusion matrix', width=1000, height=1000)
    fig.write_image(f"outputFiles/visualizations/{chemical}/PCC1_confusion_matrix.png")


