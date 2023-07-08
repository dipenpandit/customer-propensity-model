import streamlit as st
import pandas as pd
import base64


from utils import convert
from utils import predict 
from utils import filter

st.set_page_config(
    page_title="My webpage",
    page_icon=":tada:"
)

st.title("Customer Propensity Prediction for Travel Insurance Company ✈️🚢")
st.write("**Predict whether a customer is likely to make a purchase** :dollar:")
st.write('#### 📄CSV File format')

markdown_table = '''
| Age | GraduateOrNot | AnnualIncome | FamilyMembers | ChronicDiseases | FrequentFlyer | EverTravelledAbroad | Employment Type_Government Sector |
|-----|---------------|--------------|---------------|-----------------|----------------|---------------------|-----------------------------------|
|  34 |    1          | 1400000      |       5       |             1   |             1  |     1               |     0                             |
'''
st.markdown(markdown_table)
# Use st.file_uploader() to create a file uploader widget
file = st.file_uploader("Upload CSV", type=["csv"])

if st.button("Predict"):
    # Check if a file is uploaded
    try:
        if file is not None:
            # Read the uploaded file as Pandas DataFrame
            df = pd.read_csv(file)
            # Prediction 
            x_input = convert(df)
            yhat = predict(x_input)
            # Filter
            out_df = filter(yhat,df)

        # Display the DataFrame
        # st.table(out_df)
        st.table(out_df)
        out_csv = out_df.to_csv(index=False)
        # Generate a download link for the DataFrame as a CSV file
        b64 = base64.b64encode(out_csv.encode()).decode()  # Convert DataFrame to base64 string

        # Display the download link
        button_text = 'Download CSV file'

        # Create a download link for the CSV file
        href = f'data:file/csv;base64,{b64}'
        st.download_button(label=button_text, data=href, file_name='output.csv')

    except:
        st.write('### ⚠️ Please upload file!')