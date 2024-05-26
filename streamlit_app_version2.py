import streamlit as st
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def calculate_IQR(data,col):
    filtered_data = data
    modified_col, event_type_filter = modify_column_names(col)
    if event_type_filter:
        filtered_data = data[data['event_type'] == event_type_filter]
    # Calculate the quartiles and fences
    median = filtered_data[modified_col].median()
    Q1 = filtered_data[modified_col].quantile(0.25)
    Q3 = filtered_data[modified_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = max(Q1 - 1.5 * IQR, 0)
    upper_fence = Q3 + 1.5 * IQR

    return median,lower_fence,upper_fence

def calculate_percentiles_method(data,col,top_percentile,bottom_percentile):
    filtered_data = data
    modified_col, event_type_filter = modify_column_names(col)
    if event_type_filter:
        filtered_data = data[data['event_type'] == event_type_filter]
    modified_col = modify_column_names(col)[0]
    median = filtered_data[modified_col].median()
    lower_fence = filtered_data[modified_col].quantile(bottom_percentile/100)
    upper_fence = filtered_data[modified_col].quantile(top_percentile/100)

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
    
    filtered_data = data
    if event_type_filter:
        filtered_data = data[data['event_type'] == event_type_filter]
    
    if event_type_filter is not None:
        st.subheader(f"Box Plot of {event_type_filter}_{modified_col}")
    else:
        st.subheader(f"Box Plot of {modified_col}")
    
    st.write("Data points showing on plot are the values outside of fences")
    
    # Create the box plot
    plot = px.box(data_frame=filtered_data, y=modified_col)
    st.plotly_chart(plot, theme="streamlit", use_container_width=True)

    median,lower_fence,upper_fence = calculate_IQR(filtered_data,modified_col)

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

def bar_chart_sum_vs_non_outliers(data, col):
    modified_col, event_type_filter = modify_column_names(col)

    filtered_data = data
    if event_type_filter:
        filtered_data = data[data['event_type'] == event_type_filter]
    
    if modified_col == 'labor_duration':
        filtered_data = filtered_data[filtered_data[modified_col] > 0]
    if event_type_filter:
        st.subheader(f"Bar Chart of {event_type_filter}_{modified_col}, comparing with and without outliers values")
    else:
        st.subheader(f"Bar Chart of {modified_col},comparing with and without outliers values")

    # Calculate the quartiles and fences
    Q1 = filtered_data[modified_col].quantile(0.25)
    Q3 = filtered_data[modified_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = max(Q1 - 1.5 * IQR, 0)
    upper_fence = Q3 + 1.5 * IQR

    # Sum of all values
    sum_all_values = filtered_data[modified_col].sum()

    # Sum of non-outlier values
    non_outliers = filtered_data[(filtered_data[modified_col] >= lower_fence) & (filtered_data[modified_col] <= upper_fence)]
    sum_non_outlier_values = non_outliers[modified_col].sum()

    # Prepare filtered_data for Plotly
    filtered_data = {
        'Category': ['Sum of All Values', 'Sum of Non-Outlier Values'],
        str(modified_col): [sum_all_values, sum_non_outlier_values]
    }

    # Create a DataFrame
    df = pd.DataFrame(filtered_data)

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
    # Extract year from the date column
    data['year'] = pd.to_datetime(data['visit_date']).dt.year

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

def stacked_sum_graph(data, columns):
    # Extract year from the date column
    data['year'] = pd.to_datetime(data['visit_date']).dt.year

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

def count_bar_chart(data, column):
    # Extract year from the date column
    data['year'] = pd.to_datetime(data['visit_date']).dt.year

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
    st.write('Please select the wanted functions, the script will analyze only service events.')
    df = pd.read_csv(uploaded_file, low_memory=False)
    full_df = df.copy()
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

    filter_columns = st.multiselect("Select columns for **manually** filtering by setting up a maximum value", options=columns)
    filter_values = {}

    # Display selected columns and values for each selected column
    for col in filter_columns:
        modified_col, event_type_filter = modify_column_names(col)
        if modified_col == 'labor_duration':
            filter_values[col] = st.number_input(f"Enter maximum value for {event_type_filter}_{modified_col}", min_value=0.0)
            filtered_count = df[(df[modified_col] > filter_values[col]) & (df['event_type'] == event_type_filter)].shape[0]
        else:
            filter_values[col] = st.number_input(f"Enter maximum value for {modified_col}", min_value=0.0)
            filtered_count = df[df[modified_col] > filter_values[col]].shape[0]
        st.write(f"Filtered out rows: {filtered_count}")

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
        total_filtered_percentage = round(((total_rows - filtered_df.shape[0]) / total_rows) * 100,2)
        st.subheader("Total Filtered:")
        st.write(f"{(total_rows - filtered_df.shape[0])} rows were filtered out")
        st.write(f"{total_filtered_percentage}% of total rows were filtered out")

    automatic_filter_columns = st.multiselect("Select columns for **automatic** filtering", options=columns)
    automatic_filter_values = {}

    for col in automatic_filter_columns:
        automatic_filter_values[col] = None

    # Display buttons for automatic filtering actions
    if automatic_filter_columns:
        st.subheader("Automatic Filtering Information:")
        info_table_data = []

        for col in automatic_filter_columns:
            modified_col, event_type_filter = modify_column_names(col)
            filtered_df_copy = filtered_df.copy()
            if event_type_filter:
                filtered_df_copy = filtered_df[filtered_df['event_type'] == event_type_filter]

            if chosen_method == 'Interquartile Range':
                median,lower_fence,upper_fence = calculate_IQR(filtered_df_copy,modified_col)
            if chosen_method == 'Percentile Based':
                median,lower_fence,upper_fence = calculate_percentiles_method(filtered_df_copy,modified_col,top_percentile,bottom_percentile)
            formatted_median = "{:.0f}".format(median)


            # Count outliers
            num_above_fence = filtered_df_copy[filtered_df_copy[modified_col] > upper_fence].shape[0]
            formatted_num_above_fence= "{:,.0f}".format(num_above_fence)
            num_below_fence = filtered_df_copy[filtered_df_copy[modified_col] < lower_fence].shape[0]
            formatted_num_below_fence= "{:,.0f}".format(num_below_fence)

            sum_above_fence = filtered_df_copy[filtered_df_copy[modified_col] > upper_fence]['total_labor_cost'].sum()
            formatted_sum_above_fence= "{:,.0f}".format(sum_above_fence)

            sum_below_fence = filtered_df_copy[filtered_df_copy[modified_col] < lower_fence]['total_labor_cost'].sum()
            formatted_sum_below_fence= "{:,.0f}".format(sum_below_fence)

            # Calculate percentages
            total_rows = filtered_df.shape[0]
            percent_above_fence = (num_above_fence / total_rows) * 100
            percent_below_fence = (num_below_fence / total_rows) * 100

            formatted_upper_fence = "{:,.0f}".format(upper_fence)
            formatted_lower_fence = "{:,.0f}".format(lower_fence)

            # Display in table
            info_table_data.append({
                'Column': col,
                'Median': formatted_median,
                'Lower Fence Value': formatted_lower_fence,
                'Upper Fence Value': formatted_upper_fence,
                'Number Of Outliers Above Fence': formatted_num_above_fence,
                'Percentage Out Of All Data (Outliers Above Fence)': f"{percent_above_fence:.2f}%",
                'Sum of Outliers Above Fence Values' : formatted_sum_above_fence,
                'Number Of Outliers Below Fence': formatted_num_below_fence,
                'Percentage Out Of All Data (Outliers Below Fence)': f"{percent_below_fence:.2f}%",
                'Sum of Outliers Below Fence Values' : formatted_sum_below_fence})

        info_table_df = pd.DataFrame(info_table_data)
        st.table(info_table_df)

        # Continue displaying buttons for automatic filtering actions
        
        replace_with_fences_button = st.button("Replace outliers with Fences Values")
        replace_with_median_button = st.button("Replace outliers with Median")
        remove_outliers_button = st.button("Remove All Outliers")

        if remove_outliers_button or replace_with_fences_button or replace_with_median_button:
            for col in automatic_filter_columns:
                modified_col, event_type_filter = modify_column_names(col)
                filtered_df_copy = filtered_df.copy()
                if event_type_filter:
                    filtered_df_copy = filtered_df[filtered_df['event_type'] == event_type_filter]

                if chosen_method == 'Interquartile Range':
                    median,lower_fence,upper_fence = calculate_IQR(filtered_df_copy,modified_col)
                if chosen_method == 'Percentile Based':
                    median,lower_fence,upper_fence = calculate_percentiles_method(filtered_df_copy,modified_col,top_percentile,bottom_percentile)

                action = None  # Initialize action variable

                # Create a mask based on event_type_filter (if present)
                if event_type_filter:
                    mask = (filtered_df['event_type'] == event_type_filter)
                    if remove_outliers_button:
                        # Remove outliers for specific event_type_filter or all
                        filtered_df.loc[mask, modified_col] = filtered_df.loc[mask, modified_col].apply(lambda x: x if (x >= lower_fence and x <= upper_fence) else x)
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
                else:
                    if remove_outliers_button:
                    # Remove outliers for specific event_type_filter or all
                        filtered_df.loc[modified_col] = filtered_df.loc[mask, modified_col].apply(lambda x: x if (x >= lower_fence and x <= upper_fence) else x)
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

    csv_filename = f"sc_events_filtered_"
    if automatic_filter_columns:
        if remove_outliers_button or replace_with_fences_button or replace_with_median_button:
            if action:
                csv_filename+= action
                csv_filename += "_".join([f"{col}" for col in automatic_filter_values.keys()])
                

    csv_filename += "_".join([f"{col}_{value}" for col, value in filter_values.items()])
    csv_filename += ".csv"
    csv_data = filtered_df.to_csv(index=False)

    download_button_key = "_".join([f"{col}_{value}" for col, value in filter_values.items()])

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