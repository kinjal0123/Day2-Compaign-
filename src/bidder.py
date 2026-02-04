import numpy as np

class AICampaignManager:
    def __init__(self, model):
        """
        model: Saved Linear Regression model (.pkl)
        """
        self.model = model

    def suggest_actions(self, original_df, processed_df):
        # 1. Predictions
        # Kyuki aapne processor.py mein Sales_USD pehle hi hata diya hai, 
        # isliye hum direct predict kar sakte hain.
        predictions = self.model.predict(processed_df)
        
        # 2. Predicted Sales ko original data mein add karna (Negative values ko 0 kar rahe hain)
        original_df['Predicted_Sales'] = np.maximum(0, predictions)
        
        # 3. AI Bidding Logic (Decision Engine)
        def calculate_new_bid(row):
            # Case A: High Potential (Sales predict ho rahi hain aur ACoS control mein hai)
            if row['Predicted_Sales'] > 0.5 and row['ACoS_%'] < 25:
                return round(row['CPC_USD'] * 1.15, 2)  # 15% Bid Increase
            
            # Case B: Money Waster (Clicks aa rahe hain par Sales predict nahi ho rahi)
            elif row['Clicks'] > 10 and row['Predicted_Sales'] < 0.1:
                return round(row['CPC_USD'] * 0.75, 2)  # 25% Bid Decrease
            
            # Case C: Critical ACoS (Kharcha zyada ho raha hai)
            elif row['ACoS_%'] > 40:
                return round(row['CPC_USD'] * 0.85, 2)  # 15% Bid Decrease
            
            # Default: Current Bid maintain karein
            else:
                return row['CPC_USD']

        # Logic apply karke naya column banana
        original_df['AI_Suggested_Bid'] = original_df.apply(calculate_new_bid, axis=1)
        
        # Final formatting: Profitability status dikhane ke liye (Optional but looks good on UI)
        original_df['Status'] = np.where(original_df['AI_Suggested_Bid'] > original_df['CPC_USD'], 
                                         "Increasing Bid 🚀", 
                                         np.where(original_df['AI_Suggested_Bid'] < original_df['CPC_USD'], 
                                                  "Decreasing Bid 📉", "Maintain Bid ⚖️"))
        
        return original_df