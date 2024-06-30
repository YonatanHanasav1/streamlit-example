import streamlit as st
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

def calculate_IQR(data,col,event_type_column):
    IQR_data = data.copy()
    column = col
    # Calculate the quartiles and fences
    median = IQR_data[column].median()
    Q1 = IQR_data[column].quantile(0.25)
    Q3 = IQR_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = max(Q1 - 1.5 * IQR, 0)
    upper_fence = Q3 + 1.5 * IQR

    return median,lower_fence,upper_fence

def calculate_percentiles_method(data,col,top_percentile,bottom_percentile):
    percentile_data = data.copy()
    column = col
    median = percentile_data[column].median()
    lower_fence = percentile_data[column].quantile(bottom_percentile/100)
    upper_fence = percentile_data[column].quantile(top_percentile/100)

    return median,lower_fence,upper_fence


    data = data.copy()
    st.markdown(f'You are using {chosen_method} method to define outliers')

    for type in data[event_type_column].unique():
        data_copy = data.copy()
        data_copy = data[data[event_type_column] == type]

        if chosen_method == 'Interquartile Range':
            median,lower_fence,upper_fence = calculate_IQR(data_copy,column,event_type_column)
        else:
            median = calculate_IQR(data_copy,column,event_type_column)[0]
            median,lower_fence,upper_fence = calculate_percentiles_method(data_copy,column,top_percentile,bottom_percentile)

        # Display fences and median in a table
        fences_median_df = pd.DataFrame({
            'Metric': ['Upper Fence', 'Lower Fence', 'Median'],
            'Value': [upper_fence, lower_fence, median]})
        fences_median_df['Value'] = fences_median_df['Value']
        fences_median_df['Value'] = fences_median_df['Value'].map(lambda x: f"{x:,.0f}")
        st.write(f"Fences values of {type} {column}")
        st.table(fences_median_df)

        # Identify and display highest and lowest 5 outliers
        highest_5_outliers = data_copy[[column,investigation_id,event_type_column]].sort_values(by=column, ascending=False).head(5)
        col1= st.columns(1)[0]

        with col1:
            title_text = f"Highest 5 outliers for {type} {column}:"
            st.write(title_text)
            format_and_display_table(highest_5_outliers, column, investigation_id, event_type_column)

def boxplotter(column, data, chosen_method):
    st.subheader(f"Box Plot of {column}")
    st.write("Data points showing on plot are the values outside of fences")
    
    # Create the box plot for the original dataset
    if chosen_method == 'Interquartile Range':
        plot1 = px.box(data_frame=data, x=column, y='event_type', orientation='h')
    else:
        plot1 = px.box(data_frame=data, x=column, y='event_type', orientation='h')
    st.plotly_chart(plot1, theme="streamlit", use_container_width=True)

def histogram(column, data, chosen_method, event_type_column):
    fig = make_subplots(rows=2, cols=1)

    x_min = data[column].min()
    x_max = data[column].max()

    for i, event_type in enumerate(data[event_type_column].unique()):
        data_copy = data.copy()
        data_copy = data[data[event_type_column] == event_type]

        if chosen_method == 'Interquartile Range':
            median, lower_fence, upper_fence = calculate_IQR(data_copy, column, event_type_column)
        else:
            median = calculate_IQR(data_copy, column, event_type_column)[0]
            median, lower_fence, upper_fence = calculate_percentiles_method(data_copy, column, top_percentile, bottom_percentile)

        histogram_data = data_copy

        fig.add_trace(
            go.Histogram(
                x=histogram_data[column], 
                nbinsx=35, 
                name=str(event_type)  # Set the name attribute to the event type
            ),
            row=i+1, col=1
        )
        fig.add_vline(
            x=median, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f'Median: {median:.2f}', 
            annotation_position="top left", 
            row=i+1, col=1
        )
        fig.add_vline(
            x=upper_fence, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f'Upper fence: {upper_fence:.2f}', 
            annotation_position="top right", 
            row=i+1, col=1
        )

        # Update the layout to set the y-axis title for each subplot independently
        fig.update_yaxes(title_text='Number of Events', row=i+1, col=1)
        fig.update_xaxes(range=[x_min, x_max], row=i+1, col=1)

        # Adjust annotation position dynamically based on the maximum y-axis value
        max_y = max(fig.data[i]['y']) if fig.data[i]['y'] else 0
        if max_y:
            fig.update_annotations({'y': max_y}, selector=dict(text=f'Median: {median:.2f}'))

    fig.update_layout(height=850, width=600)
    st.subheader(f"Histogram of {column} by {event_type_column}")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def bar_chart_sum_vs_non_outliers(data, col, event_type_column):
    column = col

    for type in data[event_type_column].unique():
        data_copy = data.copy()
        data_copy = data[data[event_type_column] == type]
        bar_chart_data = data_copy

        if chosen_method == 'Interquartile Range':
                median,lower_fence,upper_fence = calculate_IQR(data,column,event_type_column)
        else:
            median = calculate_IQR(data,column,event_type_column)[0]
            median,lower_fence,upper_fence = calculate_percentiles_method(data,column,top_percentile,bottom_percentile)

        # Sum of all values
        sum_all_values = bar_chart_data[column].sum()
        formatted_sum_all_values = "{:,.0f}".format(sum_all_values)

        # Sum of non-outlier values
        non_outliers = bar_chart_data[(bar_chart_data[column] >= lower_fence) & (bar_chart_data[column] <= upper_fence)]
        sum_non_outlier_values = non_outliers[column].sum()
        formatted_sum_non_outlier_values = "{:,.0f}".format(sum_non_outlier_values)

        # Prepare filtered_data for Plotly
        bar_chart_data = {
            'Category': [f'Sum of all {type} values', f'Sum of non-outlier {type} values'],
            str(column): [formatted_sum_all_values, formatted_sum_non_outlier_values]}

        # Create a DataFrame
        df = pd.DataFrame(bar_chart_data)

        st.subheader(f'Sum of Values vs. Non-Outlier Values: {type} {column}')

        # Create a bar chart
        fig = px.bar(df, x='Category', y=column, color='Category', text_auto=True)

        # Display the chart using Streamlit
        st.plotly_chart(fig)

        percentage_of_change = round(100*(1-(sum_non_outlier_values / sum_all_values)),2)
        difference_of_change = round(sum_all_values - sum_non_outlier_values,2)
        formatted_difference_of_change = "{:,.0f}".format(difference_of_change)
        if 'duration' in column:
            units = 'hours'
        else:
            units = 'dollars'
        st.markdown(f'Outliers values adding up {formatted_difference_of_change} {units} which is {percentage_of_change}% out of {column} sum of values for {type} event type.')

explanation = '''In a box plot, the upper and lower fences are used to identify potential outliers in the data.
            These fences are calculated based on the interquartile range (IQR), which is a measure of statistical data scatter.
            The formula for calculating the upper and lower fences is as follows:
            Lower Fence: Q1 - K x IQR
            Upper Fence: Q3 + K x IQR
            Here: Q1 is the first quartile (25th percentile), Q3 is the third quartile (75th percentile), IQR is the interquartile range (Q3-Q1), K is a constant multiplier that determines the range beyond which data points are considered potential outliers, in our case K = 1.5.'''


st.title('Outliers Analysis')
uploaded_file = st.file_uploader(label= '', type=["csv","xlsx"])
if not uploaded_file:
    st.write('To start analysis please upload data file')

if uploaded_file:
    opening = ('The main purpose of this application is to find outliers within the dataset and filter them as needed.')
    st.write(opening)

    df = pd.read_csv(uploaded_file, low_memory=False)
    original_rows = df.shape[0]
    full_df = df.copy() # Needed for event category stacked graph
    list_of_columns = df.columns.to_list()

    st.title("Mandatory Columns Mapping")
    investigation_id = st.selectbox(label='Select investigation ID column', options= list_of_columns)
    event_type_column = st.selectbox(label='Select event_type (field / remote) column', options= list_of_columns)
    event_category = st.selectbox(label='Select event category column', options= list_of_columns)
    # Get unique values from the selected column
    unique_values = df[event_category].unique() 
    # Select multiple values from the unique values
    selected_values = st.multiselect("Select the value(s) to consider as service events", options=unique_values)
    if selected_values:
        df = df[df[event_category].isin(selected_values)]
        service_rows = df.shape[0]
        percentage = round((service_rows/original_rows)*100,2)
        formatted_original_rows = "{:,.0f}".format(original_rows)
        formatted_service_rows = "{:,.0f}".format(service_rows)
        text = f"Using {formatted_service_rows} rows out of {formatted_original_rows} rows, which is {percentage}% of total dataset."
        st.markdown(text)
    else:
        service_rows = original_rows
    data = df

    outlier_finiding_methods = ['Interquartile Range','Percentile Based']
    chosen_method = st.selectbox(label="Select outlier finding method, this will determine the outlier classification (default is IQR)", options=outlier_finiding_methods)  
    if chosen_method == 'Percentile Based':
        bottom_percentile,top_percentile = st.slider("Please select a range of values for top and bottom percentiles", 0, 100, (5,95))

    st.title("Settings")
    check_box1 = st.checkbox(label="Display a random dataset sample")
    if check_box1:
        st.subheader('Random Data Sample')
        st.write(data.sample(25))

    check_box2 = st.checkbox(label="Explain interquartile range (IQR) method")
    if check_box2:
        st.subheader('IQR outlier finding method explanation')
        lines = explanation.split('\n')
        for line in lines:
            st.write(line)

    st.title("Plots")
    #Use only the numeric columns
    list_of_columns = df.select_dtypes(include=['int', 'float']).columns
    plot_selection = st.multiselect(label="Select columns to create plots, only numeric columns are useable. The plots are splitted by event type (remote / field).", options=list_of_columns)
    
    if plot_selection:
        for col in plot_selection:
            boxplotter(col, data, chosen_method)
            histogram(col, data, chosen_method, event_type_column)
            bar_chart_sum_vs_non_outliers(data, col, event_type_column)

    st.title("Filter Data")
    st.markdown("Here you can alter the column values as needed")
    
    filter_columns = st.multiselect("Select columns for filtering", options=list_of_columns)
    filter_values = {}
    fixed_maximum_values = {}

    if filter_columns:
        st.subheader("Filtering Information:")
        info_table_data = []

        for col in filter_columns:
            df_copy = data.copy() #Used for filtering information section
            column = col
            if chosen_method == 'Interquartile Range':
                median,lower_fence,upper_fence = calculate_IQR(df_copy,column,event_type_column)
            if chosen_method == 'Percentile Based':
                median,lower_fence,upper_fence = calculate_percentiles_method(df_copy,column,top_percentile,bottom_percentile)

            formatted_median = "{:,.2f}".format(median)
            formatted_upper_fence = "{:,.2f}".format(upper_fence)
            formatted_lower_fence = "{:,.2f}".format(lower_fence)

            # Count outliers
            num_above_fence = df_copy[df_copy[column] > upper_fence].shape[0]
            formatted_num_above_fence= "{:,.0f}".format(num_above_fence)
            num_below_fence = df_copy[df_copy[column] < lower_fence].shape[0]
            formatted_num_below_fence= "{:,.0f}".format(num_below_fence)

            sum_above_fence = df_copy[df_copy[column] > upper_fence][column].sum()
            formatted_sum_above_fence= "{:,.0f}".format(sum_above_fence)

            sum_below_fence = df_copy[df_copy[column] < lower_fence][column].sum()
            formatted_sum_below_fence= "{:,.0f}".format(sum_below_fence)

            # Calculate percentages
            total_filtered_rows = df_copy.shape[0]
            if total_filtered_rows != 0:
                percent_above_fence = (num_above_fence / total_filtered_rows) * 100
                percent_below_fence = (num_below_fence / total_filtered_rows) * 100
            else:
                percent_above_fence = 0
                percent_below_fence = 0

            # Display in table
            info_table_data.append({
                'Column Name': col,
                'Median': formatted_median,
                'Lower Fence Value': formatted_lower_fence,
                'Upper Fence Value': formatted_upper_fence,
                'Number Of Outliers Above Fence': formatted_num_above_fence,
                'Percentage Out Of All Data Rows (Outliers Above Fence)': f"{percent_above_fence:.2f}%",
                'Sum of Outliers Above Fence Values' : formatted_sum_above_fence,
                'Number Of Outliers Below Fence': formatted_num_below_fence,
                'Percentage Out Of All Data Rows (Outliers Below Fence)': f"{percent_below_fence:.2f}%",
                'Sum of Outliers Below Fence Values' : formatted_sum_below_fence})

        info_table_df = pd.DataFrame(info_table_data)
        st.table(info_table_df)

    filtered_df = data.copy()

    for col in filter_columns:
        filter_values[col] = None
        # Continue displaying buttons for automatic filtering actions
        st.subheader(col)
        filtering_options_list = [f'Replace outliers with fences ({upper_fence})',f'Replace outliers with median ({median})','Choose a fixed maximum value, any value exceeding this maximum will be adjusted to match the chosen value.','Remove all outliers (choosing this options will remove the whole outlier row)']
        filter_radio = st.radio(label = 'Choose the wanted filtering method', options = filtering_options_list, key = f"{col}_filter_radio")

        if filter_radio:
            for col in filter_columns:
                column = col
                filtered_df_copy = data.copy()
                if chosen_method == 'Interquartile Range':
                    median,lower_fence,upper_fence = calculate_IQR(filtered_df_copy,column,event_type_column)
                if chosen_method == 'Percentile Based':
                    median,lower_fence,upper_fence = calculate_percentiles_method(filtered_df_copy,column,top_percentile,bottom_percentile)

                action = None  # Initialize action variable
                if filter_radio == 'Remove all outliers (choosing this options will remove the whole outlier row)':
                    # Create a mask to identify outliers
                    outlier_mask = (filtered_df[column] < lower_fence) | (filtered_df[column] > upper_fence)                        
                    # Remove rows that match the combined mask
                    filtered_df = filtered_df.drop(filtered_df[outlier_mask].index)                        
                    action = 'Remove_all_outliers_'
                    total_filtered_percentage = round(((service_rows - filtered_df.shape[0]) / service_rows) * 100,2)
                    formatted_filtered_out_rows = "{:,.0f}".format(service_rows - filtered_df.shape[0])
                    st.write(f"{(formatted_filtered_out_rows)} rows were filtered out, which are {total_filtered_percentage}% of total original file rows")
                elif filter_radio == f'Replace outliers with fences ({upper_fence})':
                    # Replace outliers with fences values 
                    filtered_df.loc[(filtered_df[column] < lower_fence), column] = lower_fence
                    filtered_df.loc[(filtered_df[column] > upper_fence), column] = upper_fence
                    action = 'Replace_outliers_with_fences_'
                elif filter_radio == f'Replace outliers with median ({median})':
                    # Replace outliers with median value 
                    filtered_df.loc[(filtered_df[column] < lower_fence), column] = median
                    filtered_df.loc[(filtered_df[column] > upper_fence), column] = median
                    action = 'Replace_outliers_with_median_'
                elif filter_radio == 'Choose a fixed maximum value, any value exceeding this maximum will be adjusted to match the chosen value.':
                    fixed_maximum_values[col] = st.number_input(f"Enter maximum fixed value for {column}",value = round(upper_fence,2))
                    filtered_df[column] = filtered_df[column].apply(lambda x: min(x, fixed_maximum_values[col]))
                    action = 'Set_fixed_maximum_value_'

        # Display the total filtered percentage above the download file button
        if filter_radio:
            see_my_changes_box = st.checkbox(label='Select this box to see how the data changed',key = f'{col} see_my_changes_box')
            if see_my_changes_box:
                for col in filter_columns:
                    boxplotter(col, filtered_df, chosen_method)
                    histogram(col, filtered_df, chosen_method, event_type_column)