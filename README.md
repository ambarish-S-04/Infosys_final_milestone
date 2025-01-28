# Risk Analysis Using Watson AI

## Project Overview

This project provides a legal and risk analysis tool using Watson AI. It allows users to upload documents (txt, pdf, csv) and receive risk and legal analyses based on the content using Watson's natural language processing capabilities. The app provides detailed risk identification, recommendations, and insights from uploaded files.

### Features:
- Upload a document (txt, pdf, csv) for analysis.
- Detailed analysis of the document focusing on legal and risk aspects.
- Actionable recommendations to mitigate risks or manage obligations.
- Integration with Watson AI for legal and risk analysis.

## Project Files:
- **Infosys_SpringBoard_Milestone_2.ipynb**: Jupyter notebook for initial experimentation and data handling.
- **LICENSE**: Contains the licensing information for the project.
- **README.md**: The readme file containing the documentation.
- **app.py**: Streamlit app code for running the risk analysis interface.
- **requirements.txt**: The file containing all necessary dependencies to run the project.
- **risk_analysis.xlsx**: Example dataset used in the project for testing purposes.

## Requirements

Before running the project, you need to install the necessary dependencies. You can install them by running:
pip install -r requirements.txt
How to Use the Project
Clone the repository:

First, clone the repository to your local machine:

bash
Copy
Edit
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
Set up the environment:

Install the required dependencies by running the following command:

bash
Copy
Edit
pip install -r requirements.txt
Configure API keys:

To use the Watson AI API, you need to configure your API key:

Go to IBM Watson AI and create a free account if you don't have one.

Once you have an account, go to the "API Keys" section and generate an API key for the Watson AI service.

Create a file named config.json in the root directory of the project with the following structure:

json
Copy
Edit
{
    "API_KEY": "your-watson-api-key",
    "URL": "your-watson-api-url"
}
Replace your-watson-api-key with your actual Watson API key and your-watson-api-url with the API URL.

Google Sheets API Integration:

If you want to use Google Sheets to store or fetch data in the project, you need to configure the Google Sheets API.

Go to the Google Developers Console.

Create a project and enable the "Google Sheets API" and "Google Drive API".

Download the credentials as a credentials.json file and place it in the root directory of the project.

Use the gspread Python library to interact with Google Sheets. You can install it via:

bash
Copy
Edit
pip install gspread oauth2client
Authenticate using the credentials file by running:

python
Copy
Edit
import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
Now you can use the Google Sheets API to read and write data to/from your Google Sheets.

Run the Streamlit app:

To start the Streamlit application, run the following command:

bash
Copy
Edit
streamlit run app.py
This will start a web application in your browser where you can upload documents and receive risk analysis.

License
This project is licensed under the MIT License - see the LICENSE file for details.
