import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

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
        st.subheader(f"Analyzing outliers of {event_type_label}_{column_str}:")
    elif column_str == 'remote_labor_duration':
        column_str = 'labor_duration'
        filtered_data = data[data['event_type'] == 'remote']
        event_type_label = 'remote'
        st.subheader(f"Analyzing outliers of {event_type_label}_{column_str}:")
    elif column_str == 'part_cost':
        column_str = 'total_part_cost'
        filtered_data = data[data['total_part_cost'] > 0]
        st.subheader(f"Analyzing outliers of {column_str}:")

    # Create boxplot using Plotly Express
    plot = px.box(data_frame=filtered_data, y=column_str)

    # Display the plot using Streamlit's `st.plotly_chart`
    st.plotly_chart(plot, theme="streamlit", use_container_width=True)

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
        if column_str == 'labor_duration':
            format_and_display_table(highest_5_outliers, title_text, column_str, 'investigation_id', 'part_name')
        elif column_str == 'total_part_cost':
            format_and_display_table(highest_5_outliers, title_text, column_str, 'part_id', 'part_name')

    with col2:
        title_text = f"Lowest 5 outliers for {event_type_label}_{column_str}:" if column_str == 'labor_duration' else f"Lowest 5 outliers for {column_str}:"
        st.write(title_text)
        if column_str == 'labor_duration':
            format_and_display_table(lowest_5_outliers, title_text, column_str, 'investigation_id', 'part_name')
        elif column_str == 'total_part_cost':
            format_and_display_table(lowest_5_outliers, title_text, column_str, 'part_id', 'part_name')
        
columns = ['field_labor_duration', 'remote_labor_duration', 'part_cost']

st.header('Outliers Analysis')
st.subheader('Getting Started')
st.write('To start analysis upload your data and select the wanted columns.')

uploaded_file = st.sidebar.file_uploader("Please load a sc_events file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)  # by default read the first sheet of the file
    data = df
    st.sidebar.title("Settings")
    check_box1 = st.sidebar.checkbox(label="Display dataset sample")

    if check_box1:
        st.write(data.head(100))

    feature_selection = st.sidebar.multiselect(label="Select columns to analyze", options=columns)
    if feature_selection:
        for col in feature_selection:
            boxplotter(col, data=df)