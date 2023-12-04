class DataFrame():
    def __init__(self,ds_path, ds_name, full_path):
        self.ds_path = 'C:/Users/God\Desktop/DieGod/Estudo/Notebooks/clusterizacao/data/raw'
        self.ds_name = 'Country-data.csv'
        self.full_path = os.path.join(ds_path, ds_name)
    
    def open(self, ds_path, ds_name, full_path):
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
        return df
    
    