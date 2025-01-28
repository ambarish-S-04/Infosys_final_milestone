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


How to Use the Project
Step 1: Clone the Repository
First, clone the repository to your local machine. Open your terminal and run:

bash
Copy
Edit
git clone https://github.com/your-username/your-repository-name.git  
cd your-repository-name
Step 2: Install Dependencies
Install the required Python dependencies listed in the requirements.txt file. Run the following command:

bash
Copy
Edit
pip install -r requirements.txt
Step 3: Configure API Keys
Watson AI API Configuration
To use the Watson AI API, follow these steps:

Create an IBM Watson Account

Go to IBM Watson AI and create a free account if you don’t already have one.
Generate an API Key

Navigate to the "API Keys" section and generate an API key for the Watson AI service.
Create a Configuration File

In the root directory of your project, create a file named config.json with the following structure:
json
Copy
Edit
{
    "API_KEY": "your-watson-api-key",
    "URL": "your-watson-api-url"
}
Replace your-watson-api-key with your actual API key and your-watson-api-url with the Watson API URL.

Step 4: Google Sheets API Integration (Optional)
Enable Google Sheets API
If you want to use Google Sheets for data storage or retrieval, configure the Google Sheets API:

Set Up a Google Cloud Project

Go to the Google Developers Console.
Create a project and enable the following APIs:
Google Sheets API
Google Drive API
Download Credentials

Download the credentials file as credentials.json.
Place it in the root directory of the project.
Install Required Libraries
Install the gspread and oauth2client libraries by running:

bash
Copy
Edit
pip install gspread oauth2client
Authenticate and Use Google Sheets
Use the following Python code snippet to authenticate and interact with Google Sheets:

python
Copy
Edit
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Define the scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Authenticate using the credentials file
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Now you can read/write data from Google Sheets
Step 5: Run the Streamlit App
To start the Streamlit web application, run the following command in your terminal:

bash
Copy
Edit
streamlit run app.py
This will launch the web application in your default browser. You can use it to upload documents and receive risk analysis.

License
This project is licensed under the MIT License. For details, refer to the LICENSE file.
