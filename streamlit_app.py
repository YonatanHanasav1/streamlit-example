import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def modify_column_names(column_str):
    if column_str == 'field_labor_duration':
        return 'labor_duration', 'field'
    elif column_str == 'remote_labor_duration':
        return 'labor_duration', 'remote'
    elif column_str == 'part_cost':
        return 'total_part_cost', None
    else:
        return column_str, None

def format_and_display_table(data, title, column_str, id_column):
    formatted_table = data.reset_index(drop=True)
    formatted_table.index += 1
    formatted_table[column_str] = formatted_table[column_str].round(2)
    formatted_table[column_str] = formatted_table[column_str].astype(str).str.replace(r'[\(\)]', '')
    formatted_table[id_column] = formatted_table[id_column].astype(str)
    formatted_table = formatted_table[[id_column, column_str]]
    st.table(formatted_table)

def boxplotter(column_str, data):
    modified_col, event_type_filter = modify_column_names(column_str)
    
    filtered_data = data
    if event_type_filter:
        filtered_data = data[data['event_type'] == event_type_filter]
    
    if modified_col == 'labor_duration':
        filtered_data = filtered_data[filtered_data[modified_col] > 0]
    
    st.subheader(f"Box Plot of {event_type_filter}_{modified_col}:")
    
    st.write("Data points showing on plot are the values outside of fences")
    plot = px.box(data_frame=filtered_data, y=modified_col)
    st.plotly_chart(plot, theme="streamlit", use_container_width=True)

    median = filtered_data[modified_col].median()
    Q1 = filtered_data[modified_col].quantile(0.25)
    Q3 = filtered_data[modified_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = max(Q1 - 1.5 * IQR, 0)
    upper_fence = Q3 + 1.5 * IQR
    
    fences_median_df = pd.DataFrame({
        'Metric': ['Upper Fence', 'Lower Fence', 'Median'],
        'Value': [upper_fence, lower_fence, median]
    })
    fences_median_df['Value'] = fences_median_df['Value'].round(2)
    st.table(fences_median_df)

    highest_5_outliers = filtered_data[[modified_col, 'investigation_id']].sort_values(by=modified_col, ascending=False).head(5)
    lowest_5_outliers = filtered_data[[modified_col, 'investigation_id']].sort_values(by=modified_col, ascending=True).head(5)

    col1, col2 = st.columns(2)

    with col1:
        title_text = f"Highest 5 outliers for {event_type_filter}_{modified_col}:" if modified_col == 'labor_duration' else f"Highest 5 outliers for {modified_col}:"
        st.write(title_text)
        format_and_display_table(highest_5_outliers, title_text, modified_col, 'investigation_id')

    with col2:
        title_text = f"Lowest 5 outliers for {event_type_filter}_{modified_col}:" if modified_col == 'labor_duration' else f"Lowest 5 outliers for {modified_col}:"
        st.write(title_text)
        format_and_display_table(lowest_5_outliers, title_text, modified_col, 'investigation_id')

def histogram(column_str, data):
    modified_col, event_type_filter = modify_column_names(column_str)

    filtered_data = data
    if event_type_filter:
        filtered_data = data[data['event_type'] == event_type_filter]
    
    if modified_col == 'labor_duration':
        filtered_data = filtered_data[filtered_data[modified_col] > 0]

    st.subheader(f"Histogram of {event_type_filter}_{modified_col}:")
    plot = px.histogram(data_frame=filtered_data, x=modified_col, nbins=30)
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
    df = pd.read_csv(uploaded_file, low_memory=False)
    total_rows = df.shape[0]
    data = df
    st.sidebar.title("Settings")
    check_box2 = st.sidebar.checkbox(label="Display IQR outlier finding method explanation")
    check_box1 = st.sidebar.checkbox(label="Display dataset sample")

    st.sidebar.title("Plots")
    boxplot_selection = st.sidebar.multiselect(label="Select columns to create box plot", options=columns)
    if boxplot_selection:
        for col in boxplot_selection:
            boxplotter(col, data=df)
    
    histogram_selection = st.sidebar.multiselect(label="Select columns to create histogram", options=columns)
    if histogram_selection:
        for col in histogram_selection:
            histogram(col, data=df)
    
    st.sidebar.title("Filter Data")
    filter_columns = st.sidebar.multiselect("Select columns for filtering", options=columns)
    filter_values = {}

    # Display selected columns and values for each selected column
    for col in filter_columns:
        modified_col, event_type_filter = modify_column_names(col)
        if modified_col == 'labor_duration':
            filter_values[col] = st.sidebar.number_input(f"Enter maximum value for {event_type_filter}_{modified_col}", min_value=0.0)
        else:
            filter_values[col] = st.sidebar.number_input(f"Enter maximum value for {modified_col}", min_value=0.0)

    # Display selected columns and values
    st.sidebar.subheader("Selected Columns and Values:")
    for col in filter_columns:
        modified_col, event_type_filter = modify_column_names(col)  # Extract modified column name
        selected_value = filter_values.get(col, 0)  # Get the selected value for the original column
        if modified_col == 'labor_duration':
            st.sidebar.write(f"{event_type_filter}_{modified_col}: {selected_value}")
        else:
            st.sidebar.write(f"{modified_col}: {selected_value}")
        # Calculate and display the filtered rows
        if modified_col == 'labor_duration':
            filtered_count = df[(df[modified_col] > filter_values[col]) & (df['event_type'] == event_type_filter)].shape[0]
        else:
            filtered_count = df[df[modified_col] > filter_values[col]].shape[0]
        st.sidebar.write(f"Filtered rows: {filtered_count}")

    filtered_df = df.copy()
    sorted_columns = []

    for col, value in filter_values.items():
        modified_col, event_type_filter = modify_column_names(col)
        if event_type_filter:
            filtered_df = filtered_df[((filtered_df['event_type'] != event_type_filter) | ((filtered_df['event_type'] == event_type_filter) & (filtered_df[modified_col] <= value)))]
        else:
            filtered_df = filtered_df[filtered_df[modified_col] <= value]

        sorted_columns.append(modified_col)
        sorted_columns = set(sorted_columns)
        sorted_columns = list(sorted_columns)

    all_columns = sorted_columns + [col for col in filtered_df.columns if col not in sorted_columns]

    if len(sorted_columns) != 0 and filter_columns:
        filtered_df = filtered_df.sort_values(by=sorted_columns, ascending=False)
        st.subheader("Filtered DataFrame:")
        st.write(filtered_df[all_columns].head(100))
        # Display the total filtered percentage above the download file button
        total_filtered_percentage = ((total_rows - filtered_df.shape[0]) / total_rows) * 100
        st.sidebar.subheader("Total Filtered Percentage:")
        st.sidebar.write(f"{total_filtered_percentage:.2f}% of rows were filtered out")

    csv_filename = f"sc_events_filtered_"
    csv_filename += "_".join([f"{col}_{value}" for col, value in filter_values.items()])
    csv_filename += ".csv"
    csv_data = filtered_df.to_csv(index=False)

    download_button_key = "_".join([f"{col}_{value}" for col, value in filter_values.items()])

    st.sidebar.download_button(
        label=f"Download {csv_filename}",
        data=csv_data,
        file_name=csv_filename,
        key=f"download_button_{download_button_key}")
        
    if check_box1:
        st.subheader('Data Sample')
        st.write(data.head(100))

    if check_box2:
        st.subheader('IQR outlier finding method explanation')
        lines = explanation.split('\n')
        for line in lines:
            st.write(line)