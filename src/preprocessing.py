from sklearn.base import BaseEstimator, TransformerMixin

class TitanicAgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.master_medians = None
        self.general_medians = None
        self.global_median = None

    def fit(self, X, y=None):
        X = X.copy()
        X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        masters = X[X['Title'] == 'Master']
        self.master_medians = masters.groupby('Pclass')['Age'].median()
        self.general_medians = X.groupby(['Pclass', 'Sex'])['Age'].median()
        self.global_median = X['Age'].median()
        
        return self

    def transform(self, X):
        X = X.copy()
        X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        mask_master = (X['Age'].isnull()) & (X['Title'] == 'Master')
        X.loc[mask_master, 'Age'] = X.loc[mask_master, 'Pclass'].map(self.master_medians)
        
        mask_remaining = X['Age'].isnull()
        fill_values = X[mask_remaining].apply(
            lambda row: self.general_medians.get((row['Pclass'], row['Sex']), self.global_median), 
            axis=1
        )
        X.loc[mask_remaining, 'Age'] = fill_values
        
        X['Age'] = X['Age'].fillna(self.global_median)
        
        return X.drop(columns=['Title', 'Name'])
