import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def format_and_display_table(data, title, column_str, id_column):
    formatted_table = data.reset_index(drop=True)
    formatted_table.index += 1  # Reset index starting from 1
    formatted_table[column_str] = formatted_table[column_str].round(2)
    formatted_table[column_str] = formatted_table[column_str].astype(str).str.replace(r'[\(\)]', '')
    formatted_table[id_column] = formatted_table[id_column].astype(str)
    formatted_table = formatted_table[[id_column, column_str]]
    st.table(formatted_table)

def boxplotter(column_str, data):
    if column_str == 'field_labor_duration':
        column_str = 'labor_duration'
        filtered_data = data[data['event_type'] == 'field']
        filtered_data = filtered_data[filtered_data[column_str]>0]
        event_type_label = 'field'
        st.subheader(f"Box Plot of {event_type_label}_{column_str}:")
    elif column_str == 'remote_labor_duration':
        column_str = 'labor_duration'
        filtered_data = data[data['event_type'] == 'remote']
        filtered_data = filtered_data[filtered_data[column_str]>0]
        event_type_label = 'remote'
        st.subheader(f"Box Plot of {event_type_label}_{column_str}:")
    elif column_str == 'part_cost':
        column_str = 'total_part_cost'
        filtered_data = data[data['total_part_cost'] > 0]
        st.subheader(f"Box Plot of {column_str}:")
    else:
        filtered_data = data[['investigation_id',column_str]]
        st.subheader(f"Box Plot of {column_str}:")

    st.write("Data points showing on plot are the values outside of fences")
    # Create boxplot using Plotly Express
    plot = px.box(data_frame=filtered_data, y=column_str)
    # Display the plot using Streamlit's `st.plotly_chart`
    st.plotly_chart(plot, theme="streamlit", use_container_width=True)
    
    # Calculate quartiles, IQR and median
    median = filtered_data[column_str].median()
    Q1 = filtered_data[column_str].quantile(0.25)
    Q3 = filtered_data[column_str].quantile(0.75)
    IQR = Q3 - Q1
    # Calculate upper and lower fences
    lower_fence = Q1 - 1.5 * IQR
    if lower_fence < 0:
        lower_fence = 0
    upper_fence = Q3 + 1.5 * IQR
    # Create a DataFrame for upper, lower fences, and median
    fences_median_df = pd.DataFrame({
        'Metric': ['Upper Fence', 'Lower Fence', 'Median'],
        'Value': [upper_fence, lower_fence, median]
    })
    fences_median_df['Value'] = fences_median_df['Value'].round(2)
    # Display the table for upper, lower fences, and median
    st.table(fences_median_df)    

    # Get the value of the most extreme outliers
    highest_5_outliers = filtered_data[[column_str, 'investigation_id']].sort_values(by=column_str, ascending=False).head(5)
    lowest_5_outliers = filtered_data[[column_str, 'investigation_id']].sort_values(by=column_str, ascending=True).head(5)

    # Display the tables side by side with matching headlines
    col1, col2 = st.columns(2)

    with col1:
        title_text = f"Highest 5 outliers for {event_type_label}_{column_str}:" if column_str == 'labor_duration' else f"Highest 5 outliers for {column_str}:"
        st.write(title_text)
        format_and_display_table(highest_5_outliers, title_text, column_str, 'investigation_id')

    with col2:
        title_text = f"Lowest 5 outliers for {event_type_label}_{column_str}:" if column_str == 'labor_duration' else f"Lowest 5 outliers for {column_str}:"
        st.write(title_text)
        format_and_display_table(lowest_5_outliers, title_text, column_str, 'investigation_id')

def histogram(column_str, data):
    if column_str == 'field_labor_duration':
        column_str = 'labor_duration'
        filtered_data = data[data['event_type'] == 'field']
        filtered_data = filtered_data[filtered_data[column_str]>0]
        event_type_label = 'field'
        st.subheader(f"Histogram of {event_type_label}_{column_str}:")
    elif column_str == 'remote_labor_duration':
        column_str = 'labor_duration'
        filtered_data = data[data['event_type'] == 'remote']
        filtered_data = filtered_data[filtered_data[column_str]>0]
        event_type_label = 'remote'
        st.subheader(f"Histogram of {event_type_label}_{column_str}:")
    elif column_str == 'part_cost':
        column_str = 'total_part_cost'
        filtered_data = data[data['total_part_cost'] > 0]
        st.subheader(f"Histogram of {column_str}:")
    else:
        filtered_data = data[['investigation_id',column_str]]
        st.subheader(f"Histogram of {column_str}:")
    # Create histogram using Plotly Express
    plot = px.histogram(data_frame=filtered_data, x=column_str, nbins=30)
    # Display the plot using Streamlit's `st.plotly_chart`
    st.plotly_chart(plot, theme="streamlit", use_container_width=True)

columns = ['field_labor_duration', 'remote_labor_duration', 'total_labor_cost', 'part_cost', 'total_cost']

explanation = '''In a box plot, the upper and lower fences are used to identify potential outliers in the data.
            These fences are calculated based on the interquartile range (IQR), which is a measure of statistical data scatter.
            The formula for calculating the upper and lower fences is as follows:
            Lower Fence: Q1 - K x IQR
            Upper Fence: Q3 + K x IQR
            Here: Q1 is the first quartile (25th percentile), Q3 is the third quartile (75th percentile), IQR is the interquartile range (Q3-Q1), K is a constant multiplier that determines the range beyond which data points are considered potential outliers, in our case K = 1.5.'''

uploaded_file = st.sidebar.file_uploader("Please load a sc_events file", type=["csv"])
st.header('Outliers Analysis')

if not uploaded_file:
    st.subheader('Getting Started')
    st.write('To start analysis upload your data and select the wanted columns.')

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)  # by default read the first sheet of the file
    data = df
    st.sidebar.title("Settings")
    check_box1 = st.sidebar.checkbox(label="Display dataset sample")
    check_box2 = st.sidebar.checkbox(label="Display IQR outlier finding method explanation")

    if check_box1:
        st.subheader('Data Sample')
        st.write(data.head(100))

    if check_box2:
        st.subheader('IQR outlier finding method explanation')
        lines = explanation.split('\n')
        for line in lines:
            st.write(line)

    boxplot_selection = st.sidebar.multiselect(label="Select columns to create box plot", options=columns)
    if boxplot_selection:
        for col in boxplot_selection:
            boxplotter(col, data=df)
    
    histogram_selection = st.sidebar.multiselect(label="Select columns to create histogram", options=columns)
    if histogram_selection:
        for col in histogram_selection:
            histogram(col, data=df)