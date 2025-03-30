
import streamlit as st
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Video Game Platform Analysis", layout="wide")

#######################

st.title("1. 🎮 Video Game Platform Sales Visualization")

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

#######################

st.header("2. 🎻 Global Sales Distribution by Category")

# Merge and reshape data for Platform / Genre / Rating
violin_data = pd.concat([
    data[['Platform', 'Global_Sales']].rename(columns={'Platform': 'Category_Value'}).assign(Category='Platform'),
    data[['Genre', 'Global_Sales']].rename(columns={'Genre': 'Category_Value'}).assign(Category='Genre'),
    data[['Rating', 'Global_Sales']].rename(columns={'Rating': 'Category_Value'}).assign(Category='Rating')
])

# Category selector
category_type = st.selectbox("Select Category Type", ['Platform', 'Genre', 'Rating'])

# Filter to top 10 categories by total sales
top_categories = (
    violin_data[violin_data['Category'] == category_type]
    .groupby('Category_Value')['Global_Sales'].sum()
    .nlargest(10).index
)

filtered = violin_data[
    (violin_data['Category'] == category_type) &
    (violin_data['Category_Value'].isin(top_categories))
]

# Violin-style plot using density estimation (Altair doesn't support real violin plots)
violin_plot = alt.Chart(filtered).transform_density(
    'Global_Sales',
    as_=['Global_Sales', 'Density'],
    groupby=['Category_Value'],
    extent=[0, filtered['Global_Sales'].quantile(0.95)],
    counts=True,
    steps=200
).mark_area(orient='horizontal', opacity=0.7).encode(
    y=alt.Y('Global_Sales:Q', title='Global Sales (Millions)', scale=alt.Scale(type='log')),
    x=alt.X('Density:Q', stack='center', axis=None),
    color=alt.Color('Category_Value:N', legend=None),
    column=alt.Column('Category_Value:N', header=alt.Header(labelOrient='bottom'))
).properties(height=400, width=100)

st.altair_chart(violin_plot, use_container_width=True)


#######################

st.header("3. 📈 Platform Popularity Forecast (2017–2020)")

# Linear regression to predict market share by company
from sklearn.linear_model import LinearRegression

# Reuse previously calculated platform share data
platform_sales = data.groupby(['Year_of_Release', 'Platform'])['Global_Sales'].sum()
total_sales_per_year = data.groupby('Year_of_Release')['Global_Sales'].sum()
platform_share = platform_sales.div(total_sales_per_year, level='Year_of_Release').unstack().fillna(0)

# Group platform sales by company
company_sales_share = pd.DataFrame(index=platform_share.index)
for company, platforms in platform_groups.items():
    company_sales_share[company] = platform_share[platforms].sum(axis=1)

# Prepare linear regression model
historical_years = company_sales_share.index.values.reshape(-1, 1)
forecast_years = np.arange(2016, 2021).reshape(-1, 1)
forecast_df = pd.DataFrame()

# Predict future shares for each company
for company in company_sales_share.columns:
    model = LinearRegression().fit(historical_years, company_sales_share[company])
    predicted = model.predict(forecast_years)
    temp = pd.DataFrame({
        'Year': forecast_years.flatten(),
        'Company': company,
        'Share': predicted * 100,
        'Type': 'Predicted'
    })
    forecast_df = pd.concat([forecast_df, temp])

# Prepare historical data
historical_df = company_sales_share.reset_index().melt(id_vars='Year_of_Release', var_name='Company', value_name='Share')
historical_df['Share'] *= 100
historical_df['Type'] = 'Actual'
historical_df = historical_df.rename(columns={'Year_of_Release': 'Year'})

# Merge actual + predicted
full_df = pd.concat([historical_df, forecast_df], ignore_index=True)

# Draw chart
line_chart = alt.Chart(full_df).mark_line(point=True).encode(
    x=alt.X('Year:O'),
    y=alt.Y('Share:Q', title='Platform Sales Share (%)'),
    color='Company:N',
    strokeDash='Type:N',
    tooltip=['Year', 'Company', alt.Tooltip('Share:Q', format='.2f'), 'Type']
).properties(width=800, height=400)

st.altair_chart(line_chart, use_container_width=True)

