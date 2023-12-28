
import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=Warning)

sns.set_style('ticks')
sns.set_context('talk')

class Methods():
    '''
    Esta classe serve para armazenar as funções que usarei na preparação dos dados
    '''
    
    def __init__(self):
        pass
    
    
    def preprocessing(self, df):
        '''
        Esta função serve para pre-processar os dados
        '''
        scale = StandardScaler()
        df_scale = pd.DataFrame(scale.fit_transform(df),
                                columns = df.columns,
                                index=df.index)
        return df_scale
    
    
    def import_df(self, file_name):
        '''
        Esta função serve para importar o dataset
        '''
        data = pd.read_csv(
            os.getcwd().replace('code', 'data').replace('data_asq_EDA', 'raw').replace('/' or '//', r'\\') 
            + f'\\{file_name}'
            )
        return data
    
    
    def export_df(self, df, file_name, file_extension='csv'):
        '''
        Esta função serve para exportar o dataset
        '''
        df.to_csv(os.getcwd().replace('code', 'data').replace('data_asq_EDA','processed')+r'\\'+f'{file_name}.{file_extension}')
        return f'Arquivo salvo!'
    
    
    def dynamic_range(self, df):
        '''
        Esta função serve para mostrar a faixa dinâmica do dataset
        '''
        plt.figure(figsize=(13,6))
        sns.boxplot(df)
        return plt.show()
    
    def bp_distribution(self, df):
      for i in range(0, 8) :
        fig, ax = plt.subplots(1, 2, figsize=(9, 2))
        plt.suptitle(df.columns[i], fontsize=15)
        sns.boxplot(x=df.columns[i], data=df, ax=ax[0]), sns.histplot(df[df.columns[i]], ax=ax[1], alpha=0.5, bins=30)
      return plt.show()
    
    def correlation(self, df):
        '''
        Esta função serve para mostrar a correlação dos dados
        '''
        plt.figure(figsize=(9,6))
        sns.heatmap(df.iloc[:,1:].corr(),annot=True,cmap='seismic')