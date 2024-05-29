import streamlit as st
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def calculate_IQR(data,col):
    IQR_data = data.copy()
    modified_col, event_type_filter = modify_column_names(col)
    if event_type_filter:
        IQR_data = data[data['event_type'] == event_type_filter]
    # Calculate the quartiles and fences
    median = IQR_data[modified_col].median()
    Q1 = IQR_data[modified_col].quantile(0.25)
    Q3 = IQR_data[modified_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = max(Q1 - 1.5 * IQR, 0)
    upper_fence = Q3 + 1.5 * IQR

    return median,lower_fence,upper_fence

def calculate_percentiles_method(data,col,top_percentile,bottom_percentile):
    percentile_data = data.copy()
    modified_col, event_type_filter = modify_column_names(col)
    if event_type_filter:
        percentile_data = data[data['event_type'] == event_type_filter]
    modified_col = modify_column_names(col)[0]
    median = percentile_data[modified_col].median()
    lower_fence = percentile_data[modified_col].quantile(bottom_percentile/100)
    upper_fence = percentile_data[modified_col].quantile(top_percentile/100)

    return median,lower_fence,upper_fence

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
    
    boxplot_data = data.copy()
    if event_type_filter:
        boxplot_data = data[data['event_type'] == event_type_filter]
    
    if event_type_filter is not None:
        st.subheader(f"Box Plot of {event_type_filter}_{modified_col}")
    else:
        st.subheader(f"Box Plot of {modified_col}")
    
    st.write("Data points showing on plot are the values outside of fences")
    
    # Create the box plot
    plot = px.box(data_frame=boxplot_data, y=modified_col)
    st.plotly_chart(plot, theme="streamlit", use_container_width=True)

    median,lower_fence,upper_fence = calculate_IQR(boxplot_data,modified_col)

    # Display fences and median in a table
    fences_median_df = pd.DataFrame({
        'Metric': ['Upper Fence', 'Lower Fence', 'Median'],
        'Value': [upper_fence, lower_fence, median]
    })
    fences_median_df['Value'] = fences_median_df['Value'].round(2)
    fences_median_df['Value'] = fences_median_df['Value'].map(lambda x: f"{x:,.2f}".rstrip('0').rstrip('.'))
    st.table(fences_median_df)

    # Identify and display highest and lowest 5 outliers
    highest_5_outliers = boxplot_data[[modified_col, 'investigation_id']].sort_values(by=modified_col, ascending=False).head(5)
    lowest_5_outliers = boxplot_data[[modified_col, 'investigation_id']].sort_values(by=modified_col, ascending=True).head(5)

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

    histogram_data = data.copy()
    if event_type_filter:
        histogram_data = data[data['event_type'] == event_type_filter]
    if event_type_filter:
        st.subheader(f"Histogram of {event_type_filter}_{modified_col}")
    else:
        st.subheader(f"Histogram of {modified_col}")
    
    median_val = histogram_data[modified_col].median()
    plot = px.histogram(data_frame=histogram_data, x=modified_col, nbins=30)
    plot.add_vline(x=median_val, line_dash="dash", line_color="red", annotation_text=f'Median: {median_val:.2f}', annotation_position="top left")

    # Update the layout to set the y-axis title
    plot.update_layout(yaxis_title='Number of Events')

    st.plotly_chart(plot, theme="streamlit", use_container_width=True)

def bar_chart_sum_vs_non_outliers(data, col):
    modified_col, event_type_filter = modify_column_names(col)

    bar_chart_data = data.copy()
    if event_type_filter:
        bar_chart_data = data[data['event_type'] == event_type_filter]
    
    # if modified_col == 'labor_duration':
    #     bar_chart_data = bar_chart_data[bar_chart_data[modified_col] > 0]
    if event_type_filter:
        st.subheader(f"Bar Chart of {event_type_filter}_{modified_col}, comparing with and without outliers values")
    else:
        st.subheader(f"Bar Chart of {modified_col},comparing with and without outliers values")

    # Calculate the quartiles and fences
    Q1 = bar_chart_data[modified_col].quantile(0.25)
    Q3 = bar_chart_data[modified_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = max(Q1 - 1.5 * IQR, 0)
    upper_fence = Q3 + 1.5 * IQR

    # Sum of all values
    sum_all_values = bar_chart_data[modified_col].sum()

    # Sum of non-outlier values
    non_outliers = bar_chart_data[(bar_chart_data[modified_col] >= lower_fence) & (bar_chart_data[modified_col] <= upper_fence)]
    sum_non_outlier_values = non_outliers[modified_col].sum()

    # Prepare filtered_data for Plotly
    bar_chart_data = {
        'Category': ['Sum of All Values', 'Sum of Non-Outlier Values'],
        str(modified_col): [sum_all_values, sum_non_outlier_values]
    }

    # Create a DataFrame
    df = pd.DataFrame(bar_chart_data)

    # Create a bar chart
    fig = px.bar(df, x='Category', y=modified_col, title=f'Sum of Values vs. Non-Outlier Values: {modified_col}', color='Category', text_auto=True)

    # Display the chart using Streamlit
    st.plotly_chart(fig)

    percentage_of_change = round(100*(1-(sum_non_outlier_values / sum_all_values)),2)
    difference_of_change = round(sum_all_values - sum_non_outlier_values,2)
    formatted_difference_of_change = "{:,.0f}".format(difference_of_change)
    if 'duration' in modified_col:
        units = 'hours'
    else:
        units = 'dollars'
    st.markdown(f'Outliers values adding up {formatted_difference_of_change} {units} which is {percentage_of_change}% out of total column sum of values')

def stacked_graph(data, column):
    # Convert 'visit_date' to datetime 
    data['visit_date'] = pd.to_datetime(data['visit_date'])
    # Extract year from the date column
    data['year'] = data['visit_date'].dt.year
    data['month'] = data['visit_date'].dt.month
    # Group by year and month and count the number of records
    monthly_counts = data.groupby(['year', 'month']).size().reset_index(name='counts')
    # Get the unique years in the dataset
    years = monthly_counts['year'].unique()

    # Group data by year and specified column and count occurrences
    stacked_data = data.groupby(['year', column]).size().unstack(fill_value=0)

    # Plot stacked bar graph using Plotly
    fig = go.Figure()
    categories = stacked_data.columns
    if column == 'event_category':
        colors = {'maintenance': 'green', 'installation': 'orange', 'service': 'blue'}
        title = 'Number of Events in each Event Cetegory Group by Year'
    elif column == 'event_type':
        colors = {'remote': 'orange', 'field': 'blue'}
        title = 'Number of Events in each Event Type Group by Year'

    for category in categories:
        fig.add_trace(go.Bar(x=stacked_data.index, y=stacked_data[category], name=category, marker_color=colors[category]))

    fig.update_layout(barmode='stack', xaxis_title="Year", yaxis_title="Number of Events",
                      title=title, xaxis=dict(tickmode='linear', tick0=min(data['year']), dtick=1),
                      uniformtext_minsize=12, uniformtext_mode='hide')

    # Add count labels inside the bars
    fig.update_traces(texttemplate='%{y:,.0f}', textposition='auto', textfont=dict(size=12, color='white', family='Arial, sans-serif'))

    st.plotly_chart(fig)

    # Loop through each year and check if all 12 months are present
    for year in years:
        year_data = monthly_counts[monthly_counts['year'] == year]
        
        if len(year_data) == 12 and set(year_data['month']) == set(range(1, 13)):
            pass
        else:
            missing_months = set(range(1, 13)) - set(year_data['month'])
            st.write(f"Please notice that year {year} is missing data for months: {sorted(missing_months)}")

def stacked_sum_graph(data, columns):
     # Convert 'visit_date' to datetime 
    data['visit_date'] = pd.to_datetime(data['visit_date'])
    # Extract year from the date column
    data['year'] = data['visit_date'].dt.year
    data['month'] = data['visit_date'].dt.month
    # Group by year and month and count the number of records
    monthly_counts = data.groupby(['year', 'month']).size().reset_index(name='counts')
    # Get the unique years in the dataset
    years = monthly_counts['year'].unique()

    # Group data by year and sum up values for each column
    summed_data = data.groupby('year')[columns].sum()

    # Plot stacked bar graph using Plotly
    fig = go.Figure()
    for column in columns:
        fig.add_trace(go.Bar(x=summed_data.index, y=summed_data[column], name=column))

    fig.update_layout(barmode='stack', xaxis_title="Year", yaxis_title="Sum of $",
                      title="Stacked Sum of Labor, Travel and Parts Costs by Year", xaxis=dict(tickmode='linear', tick0=min(data['year']), dtick=1),
                      uniformtext_minsize=12, uniformtext_mode='hide')

    # Add count labels inside the bars
    fig.update_traces(texttemplate='%{y:,.0f}', textposition='auto', textfont=dict(size=12, color='white', family='Arial, sans-serif'))

    st.plotly_chart(fig)

    # Loop through each year and check if all 12 months are present
    for year in years:
        year_data = monthly_counts[monthly_counts['year'] == year]
        
        if len(year_data) == 12 and set(year_data['month']) == set(range(1, 13)):
            pass
        else:
            missing_months = set(range(1, 13)) - set(year_data['month'])
            st.write(f"Please notice that year {year} is missing data for months: {sorted(missing_months)}")

def count_bar_chart(data, column):
    # Convert 'visit_date' to datetime 
    data['visit_date'] = pd.to_datetime(data['visit_date'])
    # Extract year from the date column
    data['year'] = data['visit_date'].dt.year
    data['month'] = data['visit_date'].dt.month
    # Group by year and month and count the number of records
    monthly_counts = data.groupby(['year', 'month']).size().reset_index(name='counts')
    # Get the unique years in the dataset
    years = monthly_counts['year'].unique()

    # Group data by year and specified column and count occurrences
    counted_data = data.groupby(['year', column]).size().reset_index(name='count')

    # Calculate the total count for each year
    total_count_by_year = counted_data.groupby('year')['count'].sum()

    # Calculate the threshold for 5% of the total count
    threshold = total_count_by_year * 0.05

    # Filter counted_data to include only rows where the count is greater than the threshold
    counted_data_filtered = counted_data[counted_data['count'] > threshold[counted_data['year']].values]

    # Plot bar chart using Plotly
    fig = px.bar(counted_data_filtered, x='year', y='count', color=column, text='count',
                 labels={'year': 'Year', 'count': 'Number of Events'}, 
                 title=f'Number of Events by each {column} by Year',
                 barmode='group')

    fig.update_layout(xaxis=dict(tickmode='linear', tick0=min(data['year']), dtick=1),
                      uniformtext_minsize=8, uniformtext_mode='show',  # Adjust label settings
                      xaxis_title='Year', yaxis_title='Number of Events',  # Label axis titles
                      bargap=0.1)  # Adjust the gap between bars

    fig.update_traces(textangle=0, textposition='auto',  # Set label position outside bars
                      texttemplate='%{y:,.0f}', textfont=dict(size=10))  # Adjust label settings

    st.plotly_chart(fig)

    # Loop through each year and check if all 12 months are present
    for year in years:
        year_data = monthly_counts[monthly_counts['year'] == year]
        
        if len(year_data) == 12 and set(year_data['month']) == set(range(1, 13)):
            pass
        else:
            missing_months = set(range(1, 13)) - set(year_data['month'])
            st.write(f"Please notice that year {year} is missing data for months: {sorted(missing_months)}")

columns = ['field_labor_duration', 'remote_labor_duration', 'travel_duration_total', 'total_labor_cost', 'part_cost']

explanation = '''In a box plot, the upper and lower fences are used to identify potential outliers in the data.
            These fences are calculated based on the interquartile range (IQR), which is a measure of statistical data scatter.
            The formula for calculating the upper and lower fences is as follows:
            Lower Fence: Q1 - K x IQR
            Upper Fence: Q3 + K x IQR
            Here: Q1 is the first quartile (25th percentile), Q3 is the third quartile (75th percentile), IQR is the interquartile range (Q3-Q1), K is a constant multiplier that determines the range beyond which data points are considered potential outliers, in our case K = 1.5.'''

uploaded_file = st.file_uploader("Please load a sc_events file", type=["csv"])
st.title('Outliers Analysis')

if not uploaded_file:
    st.write('To start analysis upload your data')

if uploaded_file:
    opening = ('''Please select the wanted functions, the script will analyze only service events. 
             Main purpose of this page is to locate outliers and filter them as needed.''')
    
    lines = opening.split('\n')
    for line in lines:
        st.write(line)

    df = pd.read_csv(uploaded_file, low_memory=False)
    full_df = df.copy() # Needed for event category stacked graph
    df = df[df['event_category'] == 'service']
    total_rows = df.shape[0]
    data = df
    
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
    plot_selection = st.multiselect(label="Select columns to create plots", options=columns)
    
    if plot_selection:
        for col in plot_selection:
            boxplotter(col, data)
            histogram(col, data)
            bar_chart_sum_vs_non_outliers(data, col)

    st.title("Filter Data")
    st.markdown("Here you can filter out values, and remove outlier rows")

    outlier_finiding_methods = ['Interquartile Range','Percentile Based']
    chosen_method = st.selectbox(label="Select outlier finding method, this will determine the outlier classification", options=outlier_finiding_methods)  
    if chosen_method == 'Percentile Based':
        bottom_percentile,top_percentile = st.slider("Please select a range of values for top and bottom percentiles", 0, 100, (5,95))
    
    filter_columns = st.multiselect("Select columns for filtering", options=columns)
    filter_values = {}
    fixed_maximum_values = {}

    if filter_columns:
        st.subheader("Filtering Information:")
        info_table_data = []

        for col in filter_columns:
            df_copy = data.copy() #Used for filtering information section
            modified_col, event_type_filter = modify_column_names(col)

            if event_type_filter:
                df_copy = df_copy[df_copy['event_type'] == event_type_filter]
            if chosen_method == 'Interquartile Range':
                median,lower_fence,upper_fence = calculate_IQR(df_copy,modified_col)
            if chosen_method == 'Percentile Based':
                median,lower_fence,upper_fence = calculate_percentiles_method(df_copy,modified_col,top_percentile,bottom_percentile)

            formatted_median = "{:,.2f}".format(median)
            formatted_upper_fence = "{:,.2f}".format(upper_fence)
            formatted_lower_fence = "{:,.2f}".format(lower_fence)

            # Count outliers
            num_above_fence = df_copy[df_copy[modified_col] > upper_fence].shape[0]
            formatted_num_above_fence= "{:,.0f}".format(num_above_fence)
            num_below_fence = df_copy[df_copy[modified_col] < lower_fence].shape[0]
            formatted_num_below_fence= "{:,.0f}".format(num_below_fence)

            sum_above_fence = df_copy[df_copy[modified_col] > upper_fence][modified_col].sum()
            formatted_sum_above_fence= "{:,.0f}".format(sum_above_fence)

            sum_below_fence = df_copy[df_copy[modified_col] < lower_fence][modified_col].sum()
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
        fixed_value_flag = False # Initialize flag variable
        # Continue displaying buttons for automatic filtering actions
        st.subheader(col)
        replace_with_fences_button = st.button("Replace outliers with Fences Values",key = str(col)+'replace_with_fences_button')
        replace_with_median_button = st.button("Replace outliers with Median",key = str(col)+'replace_with_median_button')
        remove_outliers_button = st.button("Remove All Outliers",key = str(col)+'remove_outliers_button')
        fixed_max_value_button = st.checkbox("Set Maximum Fixed Value",key = str(col)+'fixed_max_value_button')

        if remove_outliers_button or replace_with_fences_button or replace_with_median_button or fixed_max_value_button:
            for col in filter_columns:
                modified_col, event_type_filter = modify_column_names(col)
                filtered_df_copy = data.copy()
                if event_type_filter:
                    filtered_df_copy = filtered_df_copy[filtered_df_copy['event_type'] == event_type_filter]
                if chosen_method == 'Interquartile Range':
                    median,lower_fence,upper_fence = calculate_IQR(filtered_df_copy,modified_col)
                if chosen_method == 'Percentile Based':
                    median,lower_fence,upper_fence = calculate_percentiles_method(filtered_df_copy,modified_col,top_percentile,bottom_percentile)

                action = None  # Initialize action variable

                # Create a mask based on event_type_filter (if present)
                if event_type_filter:
                    mask = (filtered_df['event_type'] == event_type_filter)                    
                    if remove_outliers_button:
                        filtered_df['flag'] = 0
                        # Set the 'flag' column to 1 where the condition is met
                        filtered_df.loc[
                        (filtered_df[modified_col] > upper_fence) & 
                        (filtered_df['event_type'] == event_type_filter), 'flag'] = 1
                        filtered_df = filtered_df[filtered_df['flag']!= 1]
                        action = 'Remove_outliers_'
                    elif replace_with_fences_button:
                        # Replace outliers with fences values for specific event_type_filter or all
                        filtered_df.loc[mask & (filtered_df[modified_col] < lower_fence), modified_col] = lower_fence
                        filtered_df.loc[mask & (filtered_df[modified_col] > upper_fence), modified_col] = upper_fence
                        action = 'Replace_outliers_with_fences_'
                    elif replace_with_median_button:
                        # Replace outliers with median value for specific event_type_filter or all
                        filtered_df.loc[mask & (filtered_df[modified_col] < lower_fence), modified_col] = median
                        filtered_df.loc[mask & (filtered_df[modified_col] > upper_fence), modified_col] = median
                        action = 'Replace_outliers_with_median_'
                    elif fixed_max_value_button:
                        # Display selected columns and values for each selected column
                        fixed_maximum_values[col] = st.number_input(f"Enter maximum fixed value for {event_type_filter}_{modified_col}, minimum value set to be 0", min_value=0.0,value = round(upper_fence,2))
                        # Keep the other event type and filter the second ones
                        filtered_df = filtered_df[((filtered_df['event_type'] != event_type_filter) | ((filtered_df['event_type'] == event_type_filter) & (filtered_df[modified_col] <= fixed_maximum_values[col])))]
                        action = 'Set_fixed_maximum_value_'
                        fixed_value_flag = True                      
                else:
                    if remove_outliers_button:
                        # Create a mask to identify outliers
                        outlier_mask = (filtered_df[modified_col] < lower_fence) | (filtered_df[modified_col] > upper_fence)                        
                        # Remove rows that match the combined mask
                        filtered_df = filtered_df.drop(filtered_df[outlier_mask].index)                        
                        action = 'Remove_outliers_'
                    elif replace_with_fences_button:
                        # Replace outliers with fences values for specific event_type_filter or all
                        filtered_df.loc[(filtered_df[modified_col] < lower_fence), modified_col] = lower_fence
                        filtered_df.loc[(filtered_df[modified_col] > upper_fence), modified_col] = upper_fence
                        action = 'Replace_outliers_with_fences_'
                    elif replace_with_median_button:
                        # Replace outliers with median value for specific event_type_filter or all
                        filtered_df.loc[(filtered_df[modified_col] < lower_fence), modified_col] = median
                        filtered_df.loc[(filtered_df[modified_col] > upper_fence), modified_col] = median
                        action = 'Replace_outliers_with_median_'
                    elif fixed_max_value_button:
                        fixed_maximum_values[col] = st.number_input(f"Enter maximum fixed value for {modified_col}, minimum value set to be 0", min_value=0.0,value = round(upper_fence,2))
                        filtered_df = filtered_df[filtered_df[modified_col] <= fixed_maximum_values[col]]
                        action = 'Set_fixed_maximum_value_'
                        fixed_value_flag = True

        csv_filename = f"sc_events_filtered_"
        if filter_columns:
            if remove_outliers_button or replace_with_fences_button or replace_with_median_button:
                if action:
                    csv_filename+= action
                    csv_filename += "_".join([f"{col}" for col in filter_values.keys()])
            if fixed_value_flag:   
                    if action:
                        csv_filename+= action
                        csv_filename += "_".join([f"{col}_{round(value,2)}" for col, value in fixed_maximum_values.items()])
            csv_filename += ".csv"

            download_button_key = "_".join([f"{col}_{value}" for col, value in filter_values.items()])

        # Display the total filtered percentage above the download file button
        if remove_outliers_button or replace_with_fences_button or replace_with_median_button or fixed_value_flag:            
            total_filtered_percentage = round(((total_rows - filtered_df.shape[0]) / total_rows) * 100,2)
            st.subheader("Total Filtered:")
            formatted_filtered_out_rows = "{:,.0f}".format(total_rows - filtered_df.shape[0])
            st.write(f"{(formatted_filtered_out_rows)} rows were filtered out, which are {total_filtered_percentage}% of total original file rows")

            csv_data = filtered_df.to_csv(index=False)
            st.download_button(label=f"Download {csv_filename}", data=csv_data, file_name=csv_filename, key=f"download_button_{download_button_key}")

    # Additional Graphs
    st.title("Additional Graphs")
    EC_stacked_graph_check_box = st.checkbox(label="Display a stacked graph of event category per year")
    ET_stacked_graph_check_box = st.checkbox(label="Display a stacked graph of event type per year")
    costs_stacked_graph_check_box = st.checkbox(label="Display a stacked graph of service costs per year")
    product_type_bar_chart_check_box = st.checkbox(label="Display a bar graph of product type per year")

    if EC_stacked_graph_check_box:
        stacked_graph(full_df,'event_category')

    if ET_stacked_graph_check_box:
        stacked_graph(df,'event_type')

    if costs_stacked_graph_check_box:
        columns_to_plot = ['total_labor_cost', 'total_part_cost', 'travel_cost_total']
        stacked_sum_graph(data, columns_to_plot)

    if product_type_bar_chart_check_box:
        count_bar_chart(data,'producttype_t2')