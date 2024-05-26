import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib

#Define the function to get the color
def get_color_from_percentage(percentage, colormap='Blues'):
    normalized_value = matplotlib.colors.Normalize(vmin=0, vmax=1)(percentage)
    cmap = matplotlib.cm.get_cmap(colormap)
    return cmap(normalized_value)

# Update the function for drawing Gantt chart
def plot_gantt_with_today_marker(df, today):
    fig, ax = plt.subplots(figsize=(10, 6))
    projects = df['Project'].unique()
    project_dict = {project: i for i, project in enumerate(projects)}
    
    # Add today’s date red line on the chart
    ax.axvline(today, color='red', linestyle='-', linewidth=2, label='Today')

    for _, row in df.iterrows():
        start = row['Beginning date']
        end = row['Endding date']
        project = row['Project']
        finish_percentage = row['Finish Percentage']
        color = get_color_from_percentage(finish_percentage)
        ax.plot([start, end], [project_dict[project], project_dict[project]], marker='o', linewidth=2, color=color)
        ax.text(start + (end-start)/2, project_dict[project], f'{finish_percentage*100:.0f}%', verticalalignment='bottom', horizontalalignment='center', color='black')
    
    ax.set_yticks(range(len(projects)))
    ax.set_yticklabels([f'Project {project}' for project in projects])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=60)
    plt.xlabel('Time')
    plt.ylabel('Projects')
    plt.title('Project Gantt Chart with Finish Percentage Colors')
    plt.tight_layout()
    plt.legend(loc='upper right')
    return fig

# Streamlit application starts
st.title('Dashboard for Project Management')

# Fixed file path
file_path = 'Book2.xlsx'
df = pd.read_excel(file_path)

# Select items to display
selected_projects = st.multiselect('Select projects to display', options=df['Project'].unique(), default=df['Project'].unique()[:10])
if selected_projects:
    # Filter selected items
    filtered_df = df[df['Project'].isin(selected_projects)]
   # Draw a Gantt chart, and set today’s date to June 9, 2023
    today_date = datetime(2023, 6, 9)
    st.pyplot(plot_gantt_with_today_marker(filtered_df, today_date))
else:
    st.write("Please select at least one project to display the Gantt chart.")
    
# Load the datasets with Streamlit's experimental_memo decorator
@st.cache
def load_data(file_path):
    return pd.read_excel(file_path)

# Preprocessing functions
# Function to preprocess data for the first chart
def preprocess_data_for_chart(df1, project):
    # Filter data for the selected project
    df1_project = df1[df1['Project'] == project]
    
    # Calculate the total spending time for each category
    categories = ['A', 'B', 'C', 'D', 'E']
    total_spending = {cat: 0 for cat in categories}
    pattern = re.compile(r'([ABCDE]):(\d+\.?\d*)')

    # Sum the spending time for each category
    for _, row in df1_project.iterrows():
        matches = pattern.findall(row['Spending Time this week'])
        for cat, time in matches:
            total_spending[cat] += float(time)

    return total_spending

def calculate_employee_percentage(df1, category, project):
    df1_project = df1[df1['Project'] == project]
    employees = []
    percentages = []
    avg_times = []

    pattern = re.compile(r'{}:(\d+\.?\d*)'.format(category))

    for _, row in df1_project.iterrows():
        employees.append(row['Name'])
        match = pattern.search(row['Spending Time this week'])
        actual_time = float(match.group(1)) if match else 0
        avg_time_str = row['Personal Avg Past Spending time '].split(', ')
        avg_time = sum([float(t) for t in avg_time_str]) / len(avg_time_str) if avg_time_str[0] else 0
        percentages.append(actual_time)
        avg_times.append(avg_time)

    total_actual_time = sum(percentages)
    percentages = [time / total_actual_time * 100 if total_actual_time > 0 else 0 for time in percentages]

    return employees, percentages, avg_times

def calculate_past_avg_time_per_week(df2, project):
    # This function now takes the project as an argument to filter the data
    df2 = df2[df2['Project'] == project]
    past_avg_time_per_week = df2.groupby('Critical Things')['Past Avg Spending time per week'].sum().to_dict()
    return past_avg_time_per_week

# Rest of Streamlit app
st.title('Interactive Charts with Project Filter')

# Load data
df1_path = 'dash3data.xlsx'
df2_path = 'Corrected_Book4.xlsx'
df1 = load_data(df1_path)
df2 = load_data(df2_path)

# Selection of the project
project = st.selectbox('Select a Project:', options=sorted(df1['Project'].unique()))

# Recalculate the spending time for the selected project
total_spending_time = preprocess_data_for_chart(df1, project)
past_avg_time_per_week_project = calculate_past_avg_time_per_week(df2, project)

# First chart
st.subheader('Total Spending Time by Category')
fig, ax = plt.subplots(figsize=(10, 6))  
categories = ['A', 'B', 'C', 'D', 'E']
ax.plot(categories, [total_spending_time.get(cat, 0) for cat in categories], label='Actual', marker='o')
ax.plot(categories, [past_avg_time_per_week_project.get(cat, 0) for cat in categories], label='Past Avg', linestyle='--')
ax.set_title('Total Spending Time by Category')
ax.set_xlabel('Category')
ax.set_ylabel('Spending Time (hours)')
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# Second chart
selected_category = st.selectbox('Select a Category for Detailed View:', options=categories)
employees, percentages, avg_times = calculate_employee_percentage(df1, selected_category, project)

# Make sure to include any additional code needed for the second chart
# Placeholder for second chart (will be populated when a category is selected)
st.subheader('Employee Time Percentage for Selected Category')

fig, ax = plt.subplots(figsize=(10, 6)) 
ax.bar(employees, percentages, label='Actual')
ax.bar(employees, avg_times, width=0.5, alpha=0.5, label='Past Avg', linestyle='--')
ax.set_ylabel('Percentage (%)')
ax.set_title(f'Time Percentage per Employee for {selected_category}')
ax.legend()

plt.xticks(rotation=45, ha='right')  # Rotate labels and align right for better spacing

# Adjust layout to make sure everything fits well
plt.tight_layout()

st.pyplot(fig)

