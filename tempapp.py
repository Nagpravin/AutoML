# import streamlit as st
# import pandas as pd

# # Function to load CSV file
# @st.cache
# def load_data(file):
#     return pd.read_csv(file)

# # Function to select and store column
# def select_and_store_column(data):
#     column_name = st.sidebar.selectbox('Select Column', data.columns)
#     selected_column = data[column_name]
#     return selected_column

# def main():
#     st.title('CSV Column Selector')
    
#     # Upload CSV file
#     uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
#     if uploaded_file is not None:
#         data = load_data(uploaded_file)
        
#         selected_column = select_and_store_column(data)
        
#         # Display selected column
#         st.write(selected_column)
        
#         # Option to store or download selected column
#         if st.button('Download Selected Column as CSV'):
#             selected_column.to_csv('selected_column.csv', index=False)
#             st.success('Selected column saved as selected_column.csv')

# if __name__ == "__main__":
#     main()
#  #########################    working code below      *******************


# import streamlit as st
# import pandas as pd
# import subprocess
# # Function to load CSV file
# @st.cache_data
# def load_data(file):
#     return pd.read_csv(file)

# # Function to select and store columns
# def select_and_store_columns(data):
#     st.write("### Select Columns")
#     selected_columns = st.multiselect('Select Columns', data.columns)
#     selected_data = data[selected_columns]
#     return selected_data

# def main():
#     st.title('CSV Column Selector and Saver')
    
#     # Upload CSV file
#     uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
#     if uploaded_file is not None:
#         data = load_data(uploaded_file)
        
#         # Display all columns horizontally
#         st.write("### All Columns")
#         st.dataframe(data)
        
#         selected_data = select_and_store_columns(data)
        
#         # Display selected columns
#         st.write("### Selected Columns")
#         st.dataframe(selected_data)
        
#         # Option to store or download selected columns
#         if st.button('Download Selected Columns as CSV'):
#             selected_data.to_csv('selected_columns.csv', index=False)
#             st.success('Selected columns saved as selected_columns.csv')

# def launch_second_app():
# # Start the second Streamlit app in a subprocess
#     subprocess.Popen(["streamlit", "run", "app.py"])

# if st.button("Launch Second App"):
#     launch_second_app()
#     st.success("Second Streamlit app launched successfully!")
    
# if __name__ == "__main__":
#     main()


#  #########################    working code above      *******************




# new addition of ::
# Add a "Confirm" button to finalize column selection.
# Disable column selection after confirmation.
# Add a "Proceed" button to store the selected columns as a CSV and send it to another Streamlit app.
# select all option also  
# cancel all button which refreshes the page 


# import streamlit as st
# import pandas as pd
# import os
# import subprocess

# st.title("CSV Column Selector and Saver")

# # Function to load CSV file
# @st.cache_data
# def load_data(file):
#     return pd.read_csv(file)

# # Function to select and store columns
# def select_and_store_columns(data):
#     st.write("### Select Columns")
#     all_columns = data.columns.tolist()
#     all_columns = ["Select All"] + all_columns  # Prepend "Select All" option
#     selected_columns = st.multiselect('Select Columns', all_columns)
#     if "Select All" in selected_columns:
#         selected_columns = all_columns[1:]  # Exclude "Select All"
#     return selected_columns

# # Function to save selected columns as CSV
# def save_selected_columns(selected_columns):
#     selected_data = data[selected_columns]
#     selected_data.to_csv('selected_columns.csv', index=False)
#     st.success('Selected columns saved as selected_columns.csv')

# # Function to launch the second Streamlit app
# def launch_second_app():
#     subprocess.Popen(["streamlit", "run", "app.py"])

# # Upload CSV file
# uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

# # Check if "Cancel All" button is clicked
# if st.button("Cancel All"):
#     st.rerun()

# if uploaded_file is not None:
#     data = load_data(uploaded_file)
#     columns_confirmed = False

#     selected_columns = select_and_store_columns(data)

#     # Display selected columns
#     st.write("### Selected Columns")
#     st.write(selected_columns)

#     # Confirm selection
#     if st.button("Confirm"):
#         columns_confirmed = True

#     # Proceed to save and send selected columns
#     if columns_confirmed:
#         save_selected_columns(selected_columns)
        # if st.button("Proceed"):
        # # Launch the second Streamlit app
        #     launch_second_app()



import streamlit as st
import pandas as pd
import os
import subprocess
st.title("CSV Column Selector and Saver")

# Function to load CSV file
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Function to select and store columns
def select_and_store_columns(data):
    st.write("### Select Columns")
    all_columns = data.columns.tolist()
    all_columns = ["Select All"] + all_columns  # Prepend "Select All" option
    selected_columns = st.multiselect('Select Columns', all_columns)
    if "Select All" in selected_columns:
        selected_columns = all_columns[1:]  # Exclude "Select All"
    return selected_columns

# Function to save selected columns as CSV
def save_selected_columns(selected_columns, data):
    selected_data = data[selected_columns]
    selected_data.to_csv('selected_columns.csv', index=False)
    st.success('Selected columns saved as selected_columns.csv')

# Function to launch the second Streamlit app
def launch_second_app():
    subprocess.Popen(["streamlit", "run", "app.py"])

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

# Check if "Cancel All" button is clicked
if st.button("Cancel All"):
    st.rerun()

if uploaded_file is not None:
    data = load_data(uploaded_file)
    columns_confirmed = False

    selected_columns = select_and_store_columns(data)

    # Display selected columns
    st.write("### Selected Columns")
    st.write(selected_columns)

    # Confirm selection
    if st.button("Confirm"):
        columns_confirmed = True

    # Proceed to save and send selected columns
    if columns_confirmed:
        save_selected_columns(selected_columns, data)
        
    if st.button("Proceed"):
        launch_second_app()
