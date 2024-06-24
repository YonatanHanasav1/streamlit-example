import streamlit as st
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def calculate_IQR(data,col):
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

def format_and_display_table(data, title, column_str, id_column, event_type):    
    formatted_table = data.reset_index(drop=True)
    formatted_table.index += 1
    formatted_table[column_str] = formatted_table[column_str]
    formatted_table[column_str] = formatted_table[column_str].map(lambda x: "{:,.0f}".format(x))
    formatted_table[id_column] = formatted_table[id_column].astype(str)
    formatted_table[event_type] = formatted_table[event_type].astype(str)
    formatted_table = formatted_table[[id_column, column_str,event_type]]
    st.table(formatted_table)

def data_details(column_str, data, chosen_method,event_type):
    column = column_str
    data = data.copy()

    st.markdown(f'You are using {chosen_method} method to define outliers')

    for type in data['event_type'].unique():
        data_copy = data.copy()
        data_copy = data[data['event_type'] == type]

        if chosen_method == 'Interquartile Range':
            median,lower_fence,upper_fence = calculate_IQR(data_copy,column)
        else:
            median = calculate_IQR(data_copy,column)[0]
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
        highest_5_outliers = data_copy[[column,investigation_id,event_type]].sort_values(by=column, ascending=False).head(5)
        col1= st.columns(1)[0]

        with col1:
            title_text = f"Highest 5 outliers for {type} {column}:"
            st.write(title_text)
            format_and_display_table(highest_5_outliers, title_text, column, investigation_id, event_type)

def boxplotter(column_str, data, chosen_method):
    column = column_str
    
    boxplot_data = data.copy()
    st.subheader(f"Box Plot of {column}")
    
    st.write("Data points showing on plot are the values outside of fences")
    
    # Create the box plot
    if chosen_method == 'Interquartile Range':
        plot = px.box(data_frame=boxplot_data, x=column, y='event_type')
        st.plotly_chart(plot, theme="streamlit", use_container_width=True)
    
    else:
        median,lower_fence,upper_fence = calculate_percentiles_method(data,column,top_percentile,bottom_percentile)
        plot = px.box(data_frame=boxplot_data, x=column, y='event_type')
        st.plotly_chart(plot, theme="streamlit", use_container_width=True)

def histogram(column_str, data):
    column = column_str

    if chosen_method == 'Interquartile Range':
        median,lower_fence,upper_fence = calculate_IQR(data,column)
    else:
        median = calculate_IQR(data,column)[0]
        median,lower_fence,upper_fence = calculate_percentiles_method(data,column,top_percentile,bottom_percentile)

    for type in data['event_type'].unique():
        data_copy = data.copy()
        data_copy = data[data['event_type'] == type]

        histogram_data = data_copy
        st.subheader(f"Histogram of {type} {column}")
        
        plot = px.histogram(data_frame=histogram_data, x=column, nbins=30)
        plot.add_vline(x=median, line_dash="dash", line_color="red", annotation_text=f'Median: {median:.2f}', annotation_position="top left")
        plot.add_vline(x=upper_fence, line_dash="dash", line_color="red", annotation_text=f'Upper fence: {upper_fence:.2f}', annotation_position="top right")
        
        # Update the layout to set the y-axis title
        plot.update_layout(yaxis_title='Number of Events')

        st.plotly_chart(plot, theme="streamlit", use_container_width=True)

def bar_chart_sum_vs_non_outliers(data, col):
    column = col

    for type in data['event_type'].unique():
        data_copy = data.copy()
        data_copy = data[data['event_type'] == type]
        bar_chart_data = data_copy
        
        st.subheader(f"Bar Chart of {type} {column}, comparing with and without outliers values")

        if chosen_method == 'Interquartile Range':
                median,lower_fence,upper_fence = calculate_IQR(data,column)
        else:
            median = calculate_IQR(data,column)[0]
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
            'Category': ['Sum of All Values', 'Sum of Non-Outlier Values'],
            str(column): [formatted_sum_all_values, formatted_sum_non_outlier_values]}

        # Create a DataFrame
        df = pd.DataFrame(bar_chart_data)

        # Create a bar chart
        fig = px.bar(df, x='Category', y=column, title=f'Sum of Values vs. Non-Outlier Values: {column}', color='Category', text_auto=True)

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

explanation = '''In a box plot, the upper and lower fences are used to identify potential outliers in the data.
            These fences are calculated based on the interquartile range (IQR), which is a measure of statistical data scatter.
            The formula for calculating the upper and lower fences is as follows:
            Lower Fence: Q1 - K x IQR
            Upper Fence: Q3 + K x IQR
            Here: Q1 is the first quartile (25th percentile), Q3 is the third quartile (75th percentile), IQR is the interquartile range (Q3-Q1), K is a constant multiplier that determines the range beyond which data points are considered potential outliers, in our case K = 1.5.'''

uploaded_file = st.file_uploader(label= '', type=["csv","xlsx"])
st.title('Outliers Analysis')

if not uploaded_file:
    st.write('To start analysis please upload data file')

if uploaded_file:
    opening = ('The main purpose of this application is to find outliers within the dataset and filter them as needed.')
    st.write(opening)

    df = pd.read_csv(uploaded_file, low_memory=False)
    original_rows = df.shape[0]
    full_df = df.copy() # Needed for event category stacked graph
    list_of_columns = df.columns.to_list()

    st.title("Settings")
    investigation_id = st.selectbox(label='Select investigation ID column', options= list_of_columns)
    event_type = st.selectbox(label='Select event_type (field / remote) column', options= list_of_columns)
    if event_type:
        st.write(event_type)
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
    chosen_method = st.selectbox(label="Select outlier finding method, this will determine the outlier classification, default is IQR", options=outlier_finiding_methods)  
    if chosen_method == 'Percentile Based':
        bottom_percentile,top_percentile = st.slider("Please select a range of values for top and bottom percentiles", 0, 100, (5,95))

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
    plot_selection = st.multiselect(label="Select columns to create plots, only numeric columns are useable", options=list_of_columns)
    
    if plot_selection:
        for col in plot_selection:
            data_details(col, data, chosen_method, event_type)
            boxplotter(col, data, chosen_method)
            histogram(col, data)
            bar_chart_sum_vs_non_outliers(data, col)

    st.title("Filter Data")
    st.markdown("Here you can filter out values, and remove outlier rows")
    
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
                median,lower_fence,upper_fence = calculate_IQR(df_copy,column)
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
        fixed_value_flag = False # Initialize flag variable
        # Continue displaying buttons for automatic filtering actions
        st.subheader(col)
        replace_with_fences_button = st.button("Replace outliers with Fences Values",key = str(col)+'replace_with_fences_button')
        replace_with_median_button = st.button("Replace outliers with Median",key = str(col)+'replace_with_median_button')
        remove_outliers_button = st.button("Remove All Outliers",key = str(col)+'remove_outliers_button')
        fixed_max_value_button = st.checkbox("Set Maximum Fixed Value",key = str(col)+'fixed_max_value_button')

        if remove_outliers_button or replace_with_fences_button or replace_with_median_button or fixed_max_value_button:
            for col in filter_columns:
                column = col
                filtered_df_copy = data.copy()
                if chosen_method == 'Interquartile Range':
                    median,lower_fence,upper_fence = calculate_IQR(filtered_df_copy,column)
                if chosen_method == 'Percentile Based':
                    median,lower_fence,upper_fence = calculate_percentiles_method(filtered_df_copy,column,top_percentile,bottom_percentile)

                action = None  # Initialize action variable
                if remove_outliers_button:
                    # Create a mask to identify outliers
                    outlier_mask = (filtered_df[column] < lower_fence) | (filtered_df[column] > upper_fence)                        
                    # Remove rows that match the combined mask
                    filtered_df = filtered_df.drop(filtered_df[outlier_mask].index)                        
                    action = 'Remove_outliers_'
                elif replace_with_fences_button:
                    # Replace outliers with fences values 
                    filtered_df.loc[(filtered_df[column] < lower_fence), column] = lower_fence
                    filtered_df.loc[(filtered_df[column] > upper_fence), column] = upper_fence
                    action = 'Replace_outliers_with_fences_'
                elif replace_with_median_button:
                    # Replace outliers with median value 
                    filtered_df.loc[(filtered_df[column] < lower_fence), column] = median
                    filtered_df.loc[(filtered_df[column] > upper_fence), column] = median
                    action = 'Replace_outliers_with_median_'
                elif fixed_max_value_button:
                    fixed_maximum_values[col] = st.number_input(f"Enter maximum fixed value for {column}, minimum value set to be 0", min_value=0.0,value = round(upper_fence,2))
                    filtered_df = filtered_df[filtered_df[column] <= fixed_maximum_values[col]]
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
            total_filtered_percentage = round(((service_rows - filtered_df.shape[0]) / service_rows) * 100,2)
            st.subheader("Total Filtered:")
            formatted_filtered_out_rows = "{:,.0f}".format(service_rows - filtered_df.shape[0])
            st.write(f"{(formatted_filtered_out_rows)} rows were filtered out, which are {total_filtered_percentage}% of total original file rows")

            csv_data = filtered_df.to_csv(index=False)
            st.download_button(label=f"Download {csv_filename}", data=csv_data, file_name=csv_filename, key=f"download_button_{download_button_key}")

    # Additional Graphs
    # st.title("Additional Graphs")
    # EC_stacked_graph_check_box = st.checkbox(label="Display a stacked graph of event category per year")
    # ET_stacked_graph_check_box = st.checkbox(label="Display a stacked graph of event type per year")
    # costs_stacked_graph_check_box = st.checkbox(label="Display a stacked graph of service costs per year")
    # product_type_bar_chart_check_box = st.checkbox(label="Display a bar graph of product type per year")

    # if EC_stacked_graph_check_box:
    #     stacked_graph(full_df,'event_category')

    # if ET_stacked_graph_check_box:
    #     stacked_graph(df,'event_type')

    # if costs_stacked_graph_check_box:
    #     # columns_to_plot = ['total_labor_cost', 'total_part_cost', 'travel_cost_total']
    #     labor_cost = st.selectbox('Select labor cost column', list_of_columns)
    #     part_cost = st.selectbox('Select part cost column', list_of_columns)
    #     travel_cost = st.selectbox('Select travel cost', list_of_columns)
    #     if labor_cost != '<select>' and part_cost != '<select>' and travel_cost != '<select>':
    #         columns_to_plot = [labor_cost,part_cost,travel_cost]
    #         stacked_sum_graph(data, columns_to_plot)

    # if product_type_bar_chart_check_box:
    #     count_bar_chart(data,'producttype_t2')