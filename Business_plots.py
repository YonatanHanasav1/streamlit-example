import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def stacked_monthly_plot(df):
    # Convert visit_date to datetime
    df['visit_date'] = pd.to_datetime(df['visit_date'])

    # Extract year and month from visit_date
    df['year_month'] = df['visit_date'].dt.to_period('M')

    # Summarize the data by year_month
    monthly_summary = df.groupby('year_month').agg({
        'total_labor_cost': 'sum',
        'travel_cost_total': 'sum',
        'total_part_cost': 'sum'
    }).reset_index()

    # Convert year_month to string for plotting
    monthly_summary['year_month'] = monthly_summary['year_month'].astype(str)

    # Create the bar plot using Plotly with specified colors
    fig = px.bar(
        monthly_summary,
        x='year_month',
        y=['total_labor_cost', 'travel_cost_total', 'total_part_cost'],
        labels={'value': 'Cost (USD)', 'year_month': 'Year-Month'},
        title='Monthly Costs Breakdown',
        text_auto=True,
        color_discrete_map={
            'total_labor_cost': 'blue',
            'travel_cost_total': 'orange',
            'total_part_cost': 'green'
        }
    )

    # Customize the layout (title centered and larger, enforce a size)
    fig.update_layout(
        title={
            'text': 'Monthly Costs Breakdown',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}  
        },
        xaxis_title='Year-Month',
        yaxis_title='Cost (USD)',
        legend_title='Cost Type',
        barmode='stack'
    )

    # Display the plot in Streamlit with full container width
    st.plotly_chart(fig, use_container_width=True)

def summary_table(df):
    # Convert visit_date to datetime
    df['visit_date'] = pd.to_datetime(df['visit_date'])

    # Extract year and month from visit_date
    df['year_month'] = df['visit_date'].dt.to_period('M')

    # Summarize the data by year_month
    monthly_summary = df.groupby('year_month').agg({
        'total_labor_cost': 'sum',
        'travel_cost_total': 'sum',
        'total_part_cost': 'sum'
    }).reset_index()

    # Calculate the total cost for each month
    monthly_summary['total_cost'] = (
        monthly_summary['total_labor_cost'] +
        monthly_summary['travel_cost_total'] +
        monthly_summary['total_part_cost']
    )

    # Calculate the percentage of each cost type per month
    monthly_summary['total_labor_cost_pct'] = round(monthly_summary['total_labor_cost'] / monthly_summary['total_cost'] * 100,2)
    monthly_summary['travel_cost_total_pct'] = round(monthly_summary['travel_cost_total'] / monthly_summary['total_cost'] * 100,2)
    monthly_summary['total_part_cost_pct'] = round(monthly_summary['total_part_cost'] / monthly_summary['total_cost'] * 100,2)

    # Select only the columns with percentages and year_month
    percentage_summary = monthly_summary[['year_month', 'total_labor_cost_pct', 'travel_cost_total_pct', 'total_part_cost_pct']]

    # Convert year_month to string for readability
    percentage_summary['year_month'] = percentage_summary['year_month'].astype(str)

    # Calculate the average percentage of each cost type over the entire timeframe
    avg_labor_cost_pct = percentage_summary['total_labor_cost_pct'].mean()
    avg_travel_cost_pct = percentage_summary['travel_cost_total_pct'].mean()
    avg_part_cost_pct = percentage_summary['total_part_cost_pct'].mean()

    # Display the DataFrame with percentages
    st.write(percentage_summary)

    # Display the average percentages
    st.write("Average Percentages Over the Entire Timeframe:")
    st.write(f"Labor Cost: {avg_labor_cost_pct:.2f}%")
    st.write(f"Travel Cost: {avg_travel_cost_pct:.2f}%")
    st.write(f"Part Cost: {avg_part_cost_pct:.2f}%")

def stacked_yearly_plot(df):
    # Convert visit_date to datetime
    df['visit_date'] = pd.to_datetime(df['visit_date'])

    # Extract year from visit_date
    df['year'] = df['visit_date'].dt.year

    # Summarize the data by year
    yearly_summary = df.groupby('year').agg({
        'total_labor_cost': 'sum',
        'travel_cost_total': 'sum',
        'total_part_cost': 'sum'
    }).reset_index()

    # Calculate the total cost for each year
    yearly_summary['total_cost'] = (
        yearly_summary['total_labor_cost'] +
        yearly_summary['travel_cost_total'] +
        yearly_summary['total_part_cost']
    )

    # Convert the data to a long format for Plotly Express
    yearly_long = yearly_summary.melt(
        id_vars=['year'],
        value_vars=['total_labor_cost', 'travel_cost_total', 'total_part_cost'],
        var_name='cost_type',
        value_name='cost'
    )

    # Create a dictionary for cost type names
    cost_type_names = {
        'total_labor_cost': 'Labor Cost',
        'travel_cost_total': 'Travel Cost',
        'total_part_cost': 'Part Cost'
    }

    # Create the plot
    fig = px.bar(
        yearly_long,
        x='year',
        y='cost',
        color='cost_type',
        title='Yearly Costs Breakdown with Percentages',
        labels={'cost_type': 'Cost Type', 'cost': 'Cost (USD)'},
        text_auto='.2s',  # Display values with two significant digits
        color_discrete_map={
        'total_labor_cost': 'blue',
        'travel_cost_total': 'orange',
        'total_part_cost': 'green'
        }
    )

    # Update layout for better appearance
    fig.update_layout(
                title={
            'text': 'Yearly Costs Breakdown',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}  
        },
        xaxis_title='Year',
        yaxis_title='Cost (USD)',
        legend_title='Cost Type'
    )

    # Show the plot using Streamlit
    st.plotly_chart(fig)

def check_missing_months(df):
    # Convert visit_date to datetime if not already converted
    df['visit_date'] = pd.to_datetime(df['visit_date'])

    # Extract year and month from visit_date
    df['year'] = df['visit_date'].dt.year
    df['month'] = df['visit_date'].dt.month

    # Get the unique years in the dataset
    years = df['year'].unique()

    # Define a list of all months
    all_months = set(range(1, 13))  # Months are represented as numbers from 1 to 12

    # Initialize a dictionary to store missing months for each year
    missing_months_per_year = {}

    # Find the minimum and maximum dates
    min_date = df['visit_date'].min()
    max_date = df['visit_date'].max()

    # Calculate the difference in months and years
    total_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)
    years_covered = total_months // 12
    months_covered = total_months % 12

    # Display the total duration covered in years and months
    st.write(f"The dataset covers a period of {years_covered} years and {months_covered} months.")

    for year in years:
        # Get the months present in the data for the current year
        months_present = set(df[df['year'] == year]['month'].unique())
        
        # Find the missing months
        missing_months = all_months - months_present
        
        if missing_months:
            # If there are missing months, store them in the dictionary
            missing_months_per_year[year] = sorted(missing_months)
        else:
            # If all months are present, indicate that as well
            missing_months_per_year[year] = "All months present"

    # Display the missing months information using Streamlit
    for year, missing in missing_months_per_year.items():
        if missing == "All months present":
            st.write(f"In {year}, all months are present.")
        else:
            present_months = sorted(list(all_months - set(missing)))
            st.write(f"In {year}, present months: {present_months}, missing months: {missing}")
    
    return missing_months_per_year

st.title('Business Plots')
uploaded_file = st.file_uploader(label= '', type=["csv","xlsx"])
if not uploaded_file:
    st.write('To create the plots please upload sc_events file')

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    opening = ('The main purpose of this application is to create plots for better understanding of customer business model')
    st.write(opening)

    st.write(f"Shape of the dataset: {df.shape}")
    st.write(f"Size of the dataset: {uploaded_file.size / (1024 * 1024):.2f} MB")
    scheck_missing_months_check_box = st.checkbox(label="Display missing months information")
    if scheck_missing_months_check_box:
        check_missing_months(df)
    stacked_monthly_plot_check_box = st.checkbox(label="Display monthly costs breakdown plot")
    if stacked_monthly_plot_check_box:
        stacked_monthly_plot(df)
    summary_table_check_box = st.checkbox(label="Display monthly costs summary percentage table")
    if summary_table_check_box:
        summary_table(df)
    stacked_yearly_plot_check_box = st.checkbox(label="Display yearly costs breakdown plot")
    if stacked_yearly_plot_check_box:
        stacked_yearly_plot(df)
