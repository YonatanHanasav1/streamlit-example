import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def histogrammer(column_str, field, data, median_text=True):
    if field:
        filtered_data = data[data['event_type'] == 'field']
        event_type_label = 'field'
    else:
        filtered_data = data[data['event_type'] == 'remote']
        event_type_label = 'remote'
    
    median = round(filtered_data[column_str].median(), 1)
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(x=filtered_data[column_str],bins=20, color='blue')
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

    # Use Streamlit to display the boxplot
    st.subheader(f"Boxplot for {event_type_label}_{column_str}")
    
    # Create a boxplot using seaborn
    fig, ax = plt.subplots(figsize=(10, 2))
    sns.boxplot(x=filtered_data[column_str], fliersize=1, ax=ax)
    
    plt.xlabel(f'{event_type_label}_{column_str}')  # Add this line to include the x-axis label
    
    ax.text(0.25, 0.85, f'median={median}', color='red',
            ha="left", va="top", transform=ax.transAxes)
    
    # Display the plot using Streamlit's `st.pyplot`
    st.pyplot(fig)


def get_outliers(df,checked_column):
    print(df.head())
    df=df[df[checked_column]!=0]
    df[checked_column]=df[checked_column].astype(float)
    #histogrammer(checked_column,field=True,data = df)
    boxplotter(checked_column,field=True,data = df)

uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False) # by default read first sheet of the file
    if st.button('Find column outliers'):
            get_outliers(df,checked_column = 'labor_duration')


