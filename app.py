
import streamlit as st
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Video Game Platform Analysis", layout="wide")
st.title("🎮 Video Game Platform Sales Visualization")

@st.cache_data
def load_data():
    return pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")

data = load_data()
data = data.dropna(subset=['Year_of_Release', 'Platform', 'Global_Sales'])
data = data[data['Year_of_Release'] <= 2016]

st.header("1. Market Share of Nintendo, Sony, Microsoft Over Time")

platform_groups = {
    'Nintendo': ['Wii', 'WiiU', 'DS', '3DS', 'GC', 'NES', 'SNES', 'N64', 'GB', 'GBA'],
    'Sony': ['PS', 'PS2', 'PS3', 'PS4', 'PSP', 'PSV'],
    'Microsoft': ['XB', 'X360', 'XOne']
}

total_sales = data.groupby('Year_of_Release')['Global_Sales'].sum()
platform_sales = data.groupby(['Year_of_Release', 'Platform'])['Global_Sales'].sum()
share = platform_sales.div(total_sales, level='Year_of_Release').unstack().fillna(0)

company_df = pd.DataFrame(index=share.index)
for company, platforms in platform_groups.items():
    company_df[company] = share[platforms].sum(axis=1)

company_df = company_df.reset_index().melt(id_vars='Year_of_Release', var_name='Company', value_name='Share')
company_df['Share'] *= 100

chart = alt.Chart(company_df).mark_area(opacity=0.7).encode(
    x=alt.X('Year_of_Release:O', title='Year'),
    y=alt.Y('Share:Q', title='Market Share (%)'),
    color='Company:N',
    tooltip=['Year_of_Release', 'Company', alt.Tooltip('Share:Q', format='.2f')]
).properties(width=800, height=400)

st.altair_chart(chart, use_container_width=True)
