import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def boxplotter(column_str, field, data):
    if field:
        filtered_data = data[data['event_type'] == 'field']
        event_type_label = 'field'
    else:
        filtered_data = data[data['event_type'] == 'remote']
        event_type_label = 'remote'
    
    median = round(filtered_data[column_str].median(), 1)

    # Set Seaborn theme (you can choose different themes)
    sns.set_theme(style="whitegrid")

    # Create a boxplot using Plotly's graph_objects
    fig = go.Figure()

    # Add the box and whisker plot with only outliers
    fig.add_trace(go.Box(
        y=filtered_data[column_str],
        boxpoints="outliers",  # Show only outliers
        marker=dict(color='blue'),
        line=dict(color='blue'),
        hoverinfo='y+text',  # Display y-axis (box) and text (hovertext)
    ))

    # Add a red line for the median
    fig.add_shape(
        go.layout.Shape(
            type='line',
            x0=0,
            x1=1,
            y0=median,
            y1=median,
            line=dict(color='red', width=2)
        )
    )

    # Update layout for better aesthetics
    fig.update_layout(
        title=f"Boxplot for {event_type_label}_{column_str}",
        xaxis=dict(title=f'{event_type_label}_{column_str}'),
        showlegend=False,
        hovermode="closest",
        height=400
    )

    # Display the plot using Streamlit's `st.plotly_chart`
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False) # by default read first sheet of the file
    if st.button('Find column outliers'):
        boxplotter('labor_duration', True, data=df)
