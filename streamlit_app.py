import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def boxplotter(column_str, field, data):
    if field == 'field':
        filtered_data = data[data['event_type'] == 'field']
        event_type_label = 'field'
        st.write(f"Analyzing outliers of {event_type_label}_{column_str}:")
    elif field == 'remote':
        filtered_data = data[data['event_type'] == 'remote']
        event_type_label = 'remote'
        st.write(f"Analyzing outliers of {event_type_label}_{column_str}:")
    elif field == '':
        filtered_data = data[data['total_part_cost']>0]
        st.write(f"Analyzing outliers of {column_str}:")

    # Create a boxplot using Plotly's graph_objects
    fig = go.Figure()

    # Add the box and whisker plot with colored median line
    box_trace = go.Box(
        y=filtered_data[column_str],
        line_color='white',  # Set the color of the median line
        hoverinfo='y+text',  # Display y-axis (box) and text (hovertext)
    )
    fig.add_trace(box_trace)

    # Display the plot using Streamlit's `st.plotly_chart`
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # Get the value of the most extreme outliers
    highest_5_outliers = filtered_data[[column_str, 'investigation_id','part_name','part_id']].sort_values(by=column_str, ascending=False).head(5)
    lowest_5_outliers = filtered_data[[column_str, 'investigation_id','part_name','part_id']].sort_values(by=column_str, ascending=True).head(5)

    # Display the tables side by side with matching headlines
    col1, col2 = st.columns(2)

    if field == 'field' or field == 'remote':
        with col1:
            st.write(f"Highest 5 outliers for {event_type_label}_{column_str}:")
            formatted_highest = highest_5_outliers.reset_index(drop=True)
            formatted_highest.index += 1  # Reset index starting from 1
            formatted_highest[column_str] = formatted_highest[column_str].apply(lambda x: str(x).replace('(', '').replace(')', ''))
            formatted_highest['Investigation ID'] = formatted_highest['investigation_id'].apply(lambda x: str(x))
            formatted_highest = formatted_highest[['Investigation ID', column_str]]
            st.table(formatted_highest)

        with col2:
            st.write(f"Lowest 5 outliers for {event_type_label}_{column_str}:")
            formatted_lowest = lowest_5_outliers.reset_index(drop=True)
            formatted_lowest.index += 1  # Reset index starting from 1
            formatted_lowest[column_str] = formatted_lowest[column_str].apply(lambda x: str(x).replace('(', '').replace(')', ''))
            formatted_lowest['Investigation ID'] = formatted_lowest['investigation_id'].apply(lambda x: str(x))
            formatted_lowest = formatted_lowest[['Investigation ID', column_str]]
            st.table(formatted_lowest)
    else:
        with col1:
            st.write(f"Highest 5 outliers for {column_str}:")
            formatted_highest = highest_5_outliers.reset_index(drop=True)
            formatted_highest.index += 1  # Reset index starting from 1
            formatted_highest[column_str] = formatted_highest[column_str].apply(lambda x: str(x).replace('(', '').replace(')', ''))
            formatted_highest['Part ID'] = formatted_highest['part_id'].apply(lambda x: str(x))
            formatted_highest['Part Description'] = formatted_highest['part_name'].apply(lambda x: str(x))
            formatted_highest = formatted_highest[['Part ID','Part Description', column_str]]
            st.table(formatted_highest)

        with col2:
            st.write(f"Lowest 5 outliers for {column_str}:")
            formatted_lowest = lowest_5_outliers.reset_index(drop=True)
            formatted_lowest.index += 1  # Reset index starting from 1
            formatted_lowest[column_str] = formatted_lowest[column_str].apply(lambda x: str(x).replace('(', '').replace(')', ''))
            formatted_lowest['Part ID'] = formatted_lowest['part_id'].apply(lambda x: str(x))
            formatted_lowest['Part Description'] = formatted_lowest['part_name'].apply(lambda x: str(x))
            formatted_lowest = formatted_lowest[['Part ID','Part Description', column_str]]
            st.table(formatted_lowest)


uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False) # by default read the first sheet of the file
    if st.button('Find column outliers'):
        boxplotter('labor_duration', 'field', data=df)
        boxplotter('labor_duration', 'remote', data=df)
        boxplotter('total_part_cost', '', data=df)
