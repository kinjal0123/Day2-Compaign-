import joblib
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        # Saved files load karein
        self.scaler = joblib.load('models/scaler.pkl')
        self.train_cols = joblib.load('models/train_columns.pkl')
        
        # Scaling features list (11 features)
        self.sc_features = ["Impressions","Clicks","ACoS_%","Orders","CTR_%","CPC_USD",
                            "Conversion_Rate_%","Revenue_per_Click_USD","Day","Month","Weekday"]

    def process_input(self, df):
        # 1. Mandatory Drops: Date aur Sales_USD (Target variable) ko hatana zaroori hai
        cols_to_drop = ['Date', 'Sales_USD']
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # 2. Handling Missing Values (Bade data ke liye safety check)
        # Agar koi value missing ho toh use 0 ya mean se bhar dein
        df[self.sc_features] = df[self.sc_features].fillna(0)

        # 3. CPC Outlier Capping (Using IQR)
        # Note: Large data mein capping zaroori hai taaki model skew na ho
        q3 = df['CPC_USD'].quantile(0.75)
        q1 = df['CPC_USD'].quantile(0.25)
        upper_bound = q3 + 1.5 * (q3 - q1)
        df['CPC_USD'] = np.where(df['CPC_USD'] > upper_bound, upper_bound, df['CPC_USD'])

        # 4. One-Hot Encoding
        ohe_cols = ['Campaign_Name', 'Ad_Group', 'Match_Type', 'Keyword']
        df = pd.get_dummies(df, columns=ohe_cols, dtype=int)

        # 5. Column Alignment (Sabse important step)
        # Ye step ensure karta hai ki aapne koi bhi product filter kiya ho, 
        # structure hamesha wahi rahega jo training ke waqt tha.
        df = df.reindex(columns=self.train_cols, fill_value=0)

        # 6. Scaling
        # Transform ensures scaling is based on training data statistics
        df[self.sc_features] = self.scaler.transform(df[self.sc_features])
        
        return df