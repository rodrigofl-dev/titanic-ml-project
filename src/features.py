import pandas as pd

def engineer_features(df):
    df = df.copy()
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'U')
    
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']
    
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    
    cols_to_drop = ['Ticket', 'Cabin']

    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])
