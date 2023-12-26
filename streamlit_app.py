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
    # Checkbox for maximum value input
    st.sidebar.title("Filter Data")
    filter_columns = st.sidebar.multiselect("Select columns for filtering", options=columns)

    # Dictionary to store selected column-value pairs
    filter_values = {}

    # Display input elements for each selected column
    for col in filter_columns:
        filter_values[col] = st.sidebar.number_input(f"Enter maximum value for {col}", min_value=0.0)

    # Display selected columns and their respective values
    st.sidebar.subheader("Selected Columns and Values:")
    for col, value in filter_values.items():
        st.sidebar.write(f"{col}: {value}")

    # Apply filtering logic for selected columns and values
    filtered_df = df.copy()
    sorted_columns = []  # Initialize sorted_column list outside the loop

    for col, value in filter_values.items():
        if col == 'field_labor_duration':
            col = 'labor_duration'
            event_type_filter = 'field'
        elif col == 'remote_labor_duration':
            col = 'labor_duration'
            event_type_filter = 'remote'
        elif col == 'part_cost':
            col = 'total_part_cost'
            event_type_filter = None
        else:
            event_type_filter = None

        if event_type_filter:
            # Include rows where ['event_type'] is not equal to event_type_filter
            # OR where ['event_type'] is equal to event_type_filter and [col] is less than or equal to value
            filtered_df = filtered_df[((filtered_df['event_type'] != event_type_filter) | ((filtered_df['event_type'] == event_type_filter) & (filtered_df[col] <= value)))]

        else:
            filtered_df = filtered_df[filtered_df[col] <= value]

        # Update sorted_column inside the loop
        sorted_columns.append(col)
        sorted_columns = set(sorted_columns)
        sorted_columns = list(sorted_columns)

   # Update selected_columns outside the loop
    all_columns = sorted_columns + [col for col in filtered_df.columns if col not in sorted_columns]

    # Sort the filtered DataFrame based on the last selected column in descending order
    if len(sorted_columns) != 0 and filter_columns:
        filtered_df = filtered_df.sort_values(by=sorted_columns, ascending=False)
        # Display the filtered DataFrame with selected columns appearing first
        st.subheader("Filtered DataFrame:")
        st.write(filtered_df[all_columns].head(100))

    # Download button for filtered DataFrame as CSV
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