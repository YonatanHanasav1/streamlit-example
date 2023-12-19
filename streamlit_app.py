import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def histogrammer(column_str, field, data, median_text=True):
    if field:
        filtered_data = data[data['event_type'] == 'field']
        event_type_label = 'field'
    else:
        filtered_data = data[data['event_type'] == 'remote']
        event_type_label = 'remote'
    
    median = round(filtered_data[column_str].median(), 1)
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(x=filtered_data[column_str],bins=20, color='blue')
    plt.axvline(median, color='red', linestyle='--')
    if median_text:
        ax.text(0.25, 0.85, f'median={median}', color='red',
                ha="left", va="top", transform=ax.transAxes)
    else:
        print('Median:', median)
    plt.xlabel(f'{event_type_label}_{column_str}')  # Add this line to include the x-axis label
    plt.show()

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
        line=dict(color='blue')
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
            boxplotter(df,column_str = 'labor_duration')


