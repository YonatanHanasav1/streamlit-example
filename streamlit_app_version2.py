import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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
    formatted_table[column_str] = formatted_table[column_str].map(lambda x: "{:,.2f}".format(x))
    formatted_table[id_column] = formatted_table[id_column].astype(str)
    formatted_table = formatted_table[[id_column, column_str]]
    st.table(formatted_table)

def boxplotter(column_str, data):
    modified_col, event_type_filter = modify_column_names(column_str)
    
    filtered_data = data
    if event_type_filter:
        filtered_data = data[data['event_type'] == event_type_filter]
    
    if event_type_filter is not None:
        st.subheader(f"Box Plot of {event_type_filter}_{modified_col}")
    else:
        st.subheader(f"Box Plot of {modified_col}")
    
    st.write("Data points showing on plot are the values outside of fences")
    # Calculate the quartiles and fences
    median = filtered_data[modified_col].median()
    Q1 = filtered_data[modified_col].quantile(0.25)
    Q3 = filtered_data[modified_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = max(Q1 - 1.5 * IQR, 0)
    upper_fence = Q3 + 1.5 * IQR
    
    # Create the box plot
    plot = px.box(data_frame=filtered_data, y=modified_col)
    st.plotly_chart(plot, theme="streamlit", use_container_width=True)

    # Display fences and median in a table
    fences_median_df = pd.DataFrame({
        'Metric': ['Upper Fence', 'Lower Fence', 'Median'],
        'Value': [upper_fence, lower_fence, median]
    })
    fences_median_df['Value'] = fences_median_df['Value'].round(2)
    fences_median_df['Value'] = fences_median_df['Value'].map(lambda x: f"{x:,.2f}".rstrip('0').rstrip('.'))
    st.table(fences_median_df)

    # Identify and display highest and lowest 5 outliers
    highest_5_outliers = filtered_data[[modified_col, 'investigation_id']].sort_values(by=modified_col, ascending=False).head(5)
    lowest_5_outliers = filtered_data[[modified_col, 'investigation_id']].sort_values(by=modified_col, ascending=True).head(5)

    col1, col2 = st.columns(2)

    with col1:
        title_text = f"Highest 5 outliers for {event_type_filter}_{modified_col}:" if event_type_filter else f"Highest 5 outliers for {modified_col}:"
        st.write(title_text)
        format_and_display_table(highest_5_outliers, title_text, modified_col, 'investigation_id')

    with col2:
        title_text = f"Lowest 5 outliers for {event_type_filter}_{modified_col}:" if event_type_filter else f"Lowest 5 outliers for {modified_col}:"
        st.write(title_text)
        format_and_display_table(lowest_5_outliers, title_text, modified_col, 'investigation_id')

def histogram(column_str, data):
    modified_col, event_type_filter = modify_column_names(column_str)

    filtered_data = data
    if event_type_filter:
        filtered_data = data[data['event_type'] == event_type_filter]
    
    if modified_col == 'labor_duration':
        filtered_data = filtered_data[filtered_data[modified_col] > 0]
    if event_type_filter:
        st.subheader(f"Histogram of {event_type_filter}_{modified_col}")
    else:
        st.subheader(f"Histogram of {modified_col}")
    
    median_val = filtered_data[modified_col].median()
    plot = px.histogram(data_frame=filtered_data, x=modified_col, nbins=30)
    plot.add_vline(x=median_val, line_dash="dash", line_color="red", annotation_text=f'Median: {median_val:.2f}', annotation_position="top left")

    # Update the layout to set the y-axis title
    plot.update_layout(yaxis_title='Number of Events')

    st.plotly_chart(plot, theme="streamlit", use_container_width=True)

def pie_chart(data, col):

    modified_col = modify_column_names(col)[0]

    # Calculate fences and median
    Q1 = data[modified_col].quantile(0.25)
    Q3 = data[modified_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = max(Q1 - 1.5 * IQR, 0)
    upper_fence = Q3 + 1.5 * IQR

    # Filter the data into three groups
    below_lower_fence = data[data[modified_col] < lower_fence]
    within_fences = data[(data[modified_col] >= lower_fence) & (data[modified_col] <= upper_fence)]
    above_upper_fence = data[data[modified_col] > upper_fence]

    # Calculate the sum of each group
    sum_below_lower_fence = below_lower_fence[modified_col].sum()
    sum_within_fences = within_fences[modified_col].sum()
    sum_above_upper_fence = above_upper_fence[modified_col].sum()

    # Prepare data for Plotly
    data = {
        'Category': ['Below Lower Fence', 'Within Fences', 'Above Upper Fence'],
        'Total Part Cost': [sum_below_lower_fence, sum_within_fences, sum_above_upper_fence]}

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Create a pie chart
    fig = px.pie(df, values='Total Part Cost', names='Category', title='Sum of Total Part Cost: Outliers vs Non-Outliers')

    # Display the chart using Streamlit
    st.plotly_chart(fig)

columns = ['field_labor_duration', 'remote_labor_duration', 'travel_duration_total', 'total_labor_cost', 'part_cost']

explanation = '''In a box plot, the upper and lower fences are used to identify potential outliers in the data.
            These fences are calculated based on the interquartile range (IQR), which is a measure of statistical data scatter.
            The formula for calculating the upper and lower fences is as follows:
            Lower Fence: Q1 - K x IQR
            Upper Fence: Q3 + K x IQR
            Here: Q1 is the first quartile (25th percentile), Q3 is the third quartile (75th percentile), IQR is the interquartile range (Q3-Q1), K is a constant multiplier that determines the range beyond which data points are considered potential outliers, in our case K = 1.5.'''

uploaded_file = st.file_uploader("Please load a sc_events file", type=["csv"])
st.header('Outliers Analysis')

if not uploaded_file:
    st.write('To start analysis upload your data')

if uploaded_file:
    st.write('Please select the wanted functions using the left sidebar, the script will analyze only service events.')
    df = pd.read_csv(uploaded_file, low_memory=False)
    full_df = df.copy()
    df = df[df['event_category'] == 'service']
    total_rows = df.shape[0]
    data = df
    
    st.title("Settings")
    check_box2 = st.checkbox(label="Display IQR outlier finding method explanation")
    check_box1 = st.checkbox(label="Display a random dataset sample")

    st.title("Plots")
    plot_selection = st.multiselect(label="Select columns to create plots", options=columns)
    
    if plot_selection:
        for col in plot_selection:
            boxplotter(col, data)
            histogram(col, data)
            pie_chart(data, col)

    st.title("Filter Data")
    st.markdown("Here you can filter out values, and remove outlier rows")
    filter_columns = st.multiselect("Select columns for **manually** filtering", options=columns)
    filter_values = {}

    # Display selected columns and values for each selected column
    for col in filter_columns:
        modified_col, event_type_filter = modify_column_names(col)
        if event_type_filter:
            filtered_data = data[data['event_type'] == event_type_filter]
        else:
            filtered_data = data

        # Get min and max values
        min_val, max_val = float(filtered_data[modified_col].min()), float(filtered_data[modified_col].max())

        # Add sliders for each selected column
        filter_values[modified_col] = st.slider(f"Select range for {modified_col}", min_val, max_val, (min_val, max_val))

    # Filter data based on selected column values
    for col, (min_val, max_val) in filter_values.items():
        data = data[(data[col] >= min_val) & (data[col] <= max_val)]

    # Display random sample and explanation
    if check_box1:
        st.subheader('Random sample')
        st.write(data.sample(3))

    if check_box2:
        st.subheader('Explanation')
        st.write(explanation)

    col1, col2 = st.columns(2)
    with col1:
        st.write('Rows in uploaded file:')
        st.subheader(str(total_rows))
    with col2:
        st.write('Rows after filtering')
        st.subheader(str(data.shape[0]))

    # Additional Graphs
    st.title("Additional Graphs")
    additional_columns = st.multiselect("Select columns for additional graphs", options=columns)
    if additional_columns:
        for col in additional_columns:
            st.write(f"Additional graph for {col}:")
            # Add your additional graphs logic here
