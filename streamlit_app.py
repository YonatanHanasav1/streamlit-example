import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False) # by default read first sheet of the file

    df.head()
    
    checked_column = 'labor_duration'
    df=df[df[checked_column]!=0]
    df[checked_column]=df[checked_column].astype(float)
    
    def histogrammer(column_str, field, data, median_text=True, **kwargs):
        if field:
            filtered_data = data[data['event_type'] == 'field']
            event_type_label = 'field'
        else:
            filtered_data = data[data['event_type'] == 'remote']
            event_type_label = 'remote'
        
        median = round(filtered_data[column_str].median(), 1)
        plt.figure(figsize=(8, 5))
        ax = sns.histplot(x=filtered_data[column_str], **kwargs)
        plt.axvline(median, color='red', linestyle='--')
        if median_text:
            ax.text(0.25, 0.85, f'median={median}', color='red',
                    ha="left", va="top", transform=ax.transAxes)
        else:
            print('Median:', median)
        plt.xlabel(f'{event_type_label}_{column_str}')  # Add this line to include the x-axis label
        plt.show()
    def boxplotter(column_str, field, data):
        if field:
            filtered_data = data[data['event_type'] == 'field']
            event_type_label = 'field'
        else:
            filtered_data = data[data['event_type'] == 'remote']
            event_type_label = 'remote'
        
        median = round(filtered_data[column_str].median(), 1)
        
        plt.figure(figsize=(10, 2))
        ax = sns.boxplot(x=filtered_data[column_str], fliersize=1)
        
        plt.xlabel(f'{event_type_label}_{column_str}')  # Add this line to include the x-axis label
        
        ax.text(0.25, 0.85, f'median={median}', color='red',
                ha="left", va="top", transform=ax.transAxes)
        
        plt.show()
    
    # Calculate 25th percentile of annual strikes
    percentile25 = df[checked_column].quantile(0.25)
    
    # Calculate 75th percentile of annual strikes
    percentile75 = df[checked_column].quantile(0.75)
    
    # Calculate interquartile range
    iqr = percentile75 - percentile25
    
    # Calculate upper and lower thresholds for outliers
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    
    print('Lower limit is: ', lower_limit)
    print('Upper limit is: ', upper_limit)
    
    histogrammer(checked_column,field=True,data = df)
    
    boxplotter(checked_column,field=True,data = df)
