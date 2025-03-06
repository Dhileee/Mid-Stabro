import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import folium
from streamlit_folium import folium_static

# Load dataset
@st.cache_data
def load_data():
    file_path = "1900-2021.csv"
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Cek kolom yang ada di dataset
st.write("Columns in dataset:", df.columns)

# Sidebar filter
st.sidebar.header("Filters")
year_range = st.sidebar.slider("Select Year Range", int(df['Year'].min()), int(df['Year'].max()), (2000, 2021))
df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

# Title and intro
st.title("🌍 Natural Disaster Data Visualization 🌪")
st.markdown("*Explore the impact of natural disasters worldwide!*")

# Basic statistics
st.subheader("📊 Key Statistics")
st.write(f"*Total disaster events ({year_range[0]} - {year_range[1]}):* {len(df_filtered)}")
st.write(f"*Average affected per event:* {df_filtered['Total Affected'].mean():,.0f}")
st.write("*Probability of occurrence per disaster type:*")
st.dataframe(df_filtered['Disaster Type'].value_counts(normalize=True).rename_axis('Disaster Type').reset_index(name='Probability'))

# Bar plot: Number of events per disaster type
st.subheader("🌡 Number of Events per Disaster Type")
disaster_counts = df_filtered['Disaster Type'].value_counts()
fig = px.bar(x=disaster_counts.index, 
             y=disaster_counts.values,
             labels={'x': 'Disaster Type', 'y': 'Number of Events'},
             title='Disaster Frequency', height=500)
st.plotly_chart(fig)

# Disaster trend per year
st.subheader("📈 Natural Disaster Trend per Year")
fig = px.line(df_filtered.groupby('Year').size().reset_index(name='Count'), x='Year', y='Count',
              markers=True, title='Trend of Natural Disasters Over the Years')
st.plotly_chart(fig)

# Scatter plot: Total Affected vs. Damages
st.subheader("💥 Impact Analysis: Affected People vs. Damage")
fig = px.scatter(df_filtered, x="Total Affected", y="Total Damages ('000 US$)", color='Disaster Type',
                 size_max=60, log_x=True, log_y=True, title='Affected People vs. Economic Damage')
st.plotly_chart(fig)

# Heatmap for disasters per region
st.subheader("🌍 Heatmap: Disaster Occurrence by Region")
region_pivot = df_filtered.pivot_table(index='Region', columns='Disaster Type', aggfunc='size', fill_value=0)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(region_pivot, cmap='coolwarm', annot=True, fmt='d', linewidths=.5, ax=ax)
st.pyplot(fig)

# Word Cloud for Disaster Types
st.subheader("☁ Most Frequent Disaster Types")
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='coolwarm').generate(" ".join(df_filtered['Disaster Type']))
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# Distribution of disaster probability per region
st.subheader("📍 Probability Distribution of Disasters by Region")
fig = px.histogram(df_filtered, x='Region', color='Disaster Type', barnorm='percent',
                   title='Probability Distribution of Disasters by Region',
                   labels={'Region': 'Region', 'count': 'Percentage'}, height=500)
st.plotly_chart(fig)

# Map Visualization
st.subheader("🗺 Disaster Locations Map")
# Pastikan kolom Latitude dan Longitude numerik
df_filtered = df_filtered[pd.to_numeric(df_filtered['Latitude'], errors='coerce').notna() & pd.to_numeric(df_filtered['Longitude'], errors='coerce').notna()]
df_filtered['Latitude'] = df_filtered['Latitude'].astype(float)
df_filtered['Longitude'] = df_filtered['Longitude'].astype(float)

if not df_filtered.empty:
    map_center = [df_filtered['Latitude'].mean(), df_filtered['Longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=2)
    for _, row in df_filtered.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=f"{row['Disaster Type']} ({row['Year']})"
        ).add_to(m)
    folium_static(m)
else:
    st.write("No valid location data available for mapping.")

# Conclusion
st.subheader("📌 Conclusion")
st.markdown("- The most common type of natural disaster in the selected years is *Floods*.")
st.markdown("- The probability of different disaster types varies, with floods and storms being the most frequent.")
st.markdown("- Economic damage and total affected population show a *positive correlation* in some cases.")
st.markdown("- Geospatial analysis shows disasters are concentrated in certain regions prone to specific events.")

st.write("Data source: Kaggle")