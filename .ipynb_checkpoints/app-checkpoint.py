import streamlit as st
import pandas as pd
import joblib
from src.processor import DataProcessor
from src.bidder import AICampaignManager

st.set_page_config(page_title="AI Campaign Tool", layout="wide")
st.title("🤖 Amazon AI Campaign Profit Optimizer")

@st.cache_resource
def init_system():
    model = joblib.load('models/Model.pkl')
    processor = DataProcessor()
    manager = AICampaignManager(model)
    return processor, manager

processor, manager = init_system()

file = st.file_uploader("Upload Campaign CSV", type=['csv'])

if file:
    df = pd.read_csv(file)
    
    # --- Sidebar Filters Start ---
    st.sidebar.header("🔍 Filter & Search")
    
    # 1. Dropdown with Search (Searchable by typing)
    all_campaigns = ["All"] + sorted(df['Campaign_Name'].unique().tolist())
    selected_camp = st.sidebar.selectbox("Select or Type Campaign Name", all_campaigns)
    
    # 2. Manual Text Search (For Keyword or Product)
    search_query = st.sidebar.text_input("Manual Search (Keyword/Product Name)", "")
    # --- Sidebar Filters End ---

    st.write("### Raw Data Preview", df.head(3))
    
    if st.button("🚀 Run AI Analysis"):
        # Global Analysis (Peeche poora data analyze hoga)
        processed = processor.process_input(df.copy())
        results = manager.suggest_actions(df, processed)
        
        # Filtering Logic
        filtered_results = results.copy()
        
        # Apply Campaign Filter
        if selected_camp != "All":
            filtered_results = filtered_results[filtered_results['Campaign_Name'] == selected_camp]
        
        # Apply Manual Text Search (Case-insensitive)
        if search_query:
            # Ye Keyword aur Campaign_Name dono mein search karega
            mask = filtered_results['Keyword'].str.contains(search_query, case=False, na=False) | \
                   filtered_results['Campaign_Name'].str.contains(search_query, case=False, na=False)
            filtered_results = filtered_results[mask]
        
        # Check if results exist after filtering
        if filtered_results.empty:
            st.warning("⚠️ No data found for the selected filters.")
        else:
            # Metrics (Ab ye sirf filtered data ka sum/avg dikhayenge)
            st.subheader(f"Key Insights: {selected_camp if selected_camp != 'All' else 'Global'}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Selected Rows", len(filtered_results))
            c2.metric("Total Predicted Sales", f"${filtered_results['Predicted_Sales'].sum():.2f}")
            c3.metric("Avg. Optimized Bid", f"${filtered_results['AI_Suggested_Bid'].mean():.2f}")
            
            # Show Results Table
            st.write("### AI Optimized Report")
            display_cols = ['Campaign_Name', 'Keyword', 'Predicted_Sales', 'CPC_USD', 'AI_Suggested_Bid', 'Status']
            st.dataframe(filtered_results[display_cols], use_container_width=True)
            
            # Download Button (Sirf filtered data download hoga)
            csv = filtered_results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Filtered Report", csv, "optimized_filtered.csv", "text/csv")