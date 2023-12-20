import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def display_dataset(data):
    st.sidebar.title("Settings")
    check_box1 = st.sidebar.checkbox(label="Display dataset")

    if check_box1:
        st.write(data.head())

def format_and_display_table(data, title, column_str, id_column, name_column):
    formatted_table = data.reset_index(drop=True)
    formatted_table.index += 1  # Reset index starting from 1
    formatted_table[column_str] = formatted_table[column_str].astype(str).str.replace(r'[\(\)]', '')
    formatted_table[id_column] = formatted_table[id_column].astype(str)
    formatted_table[name_column] = formatted_table[name_column].astype(str)

    if column_str == 'labor_duration':
        formatted_table = formatted_table[[id_column, column_str]]
    else:
        formatted_table = formatted_table[[id_column, name_column, column_str]]

    st.table(formatted_table)

def boxplotter(column_str, data):
    if column_str == 'field_labor_duration':
        column_str = 'labor_duration'
        filtered_data = data[data['event_type'] == 'field']
        event_type_label = 'field'
        st.write(f"Analyzing outliers of {event_type_label}_{column_str}:")
    elif column_str == 'remote_labor_duration':
        column_str = 'labor_duration'
        filtered_data = data[data['event_type'] == 'remote']
        event_type_label = 'remote'
        st.write(f"Analyzing outliers of {event_type_label}_{column_str}:")
    elif column_str == 'part_cost':
        column_str = 'total_part_cost'
        filtered_data = data[data['total_part_cost'] > 0]
        st.write(f"Analyzing outliers of {column_str}:")

    # Create a boxplot using Plotly's graph_objects
    fig = go.Figure()

    # Add the box and whisker plot with colored median line
    box_trace = go.Box(
        y=filtered_data[column_str],
        line_color='white',  # Set the color of the median line
        hoverinfo='y+text',  # Display y-axis (box) and text (hovertext)
    )
    fig.add_trace(box_trace)

    # Display the plot using Streamlit's `st.plotly_chart`
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # Get the value of the most extreme outliers
    highest_5_outliers = filtered_data[[column_str, 'investigation_id', 'part_name', 'part_id']].sort_values(
        by=column_str, ascending=False).head(5)
    lowest_5_outliers = filtered_data[[column_str, 'investigation_id', 'part_name', 'part_id']].sort_values(
        by=column_str, ascending=True).head(5)

    # Display the tables side by side with matching headlines
    col1, col2 = st.columns(2)

    with col1:
        title_text = f"Highest 5 outliers for {event_type_label}_{column_str}:" if column_str == 'labor_duration' else f"Highest 5 outliers for {column_str}:"
        st.write(title_text)
        format_and_display_table(highest_5_outliers, title_text, column_str, 'investigation_id', 'part_name')

    with col2:
        title_text = f"Lowest 5 outliers for {event_type_label}_{column_str}:" if column_str == 'labor_duration' else f"Lowest 5 outliers for {column_str}:"
        st.write(title_text)
        format_and_display_table(lowest_5_outliers, title_text, column_str, 'investigation_id', 'part_name')

columns = ['field_labor_duration', 'remote_labor_duration', 'part_cost']
uploaded_file = st.file_uploader("Please load sc_events file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)  # by default read the first sheet of the file
    display_dataset(df)

    feature_selection = st.sidebar.multiselect(label="Select columns to analyze", options=columns)

    if feature_selection:
        for col in feature_selection:
            boxplotter(col, data=df)
