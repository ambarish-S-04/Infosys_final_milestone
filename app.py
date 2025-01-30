import os
import json
import gspread
import pandas as pd
import smtplib
import streamlit as st
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from oauth2client.service_account import ServiceAccountCredentials
from transformers import pipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

# =================== Streamlit UI =================== #
st.title("📜 Legal & Risk Analysis Tool")

st.write("Upload a text document for legal risk analysis and receive results via email.")

# File Upload
uploaded_file = st.file_uploader("📂 Upload a text file", type=["txt"])
query = st.text_input("🔍 Enter your query:", key="query_input")
credentials_file = st.text_input("📜 Enter Google Sheets credentials file path:", key="credentials_file")
sheet_name = st.text_input("📊 Enter Google Sheet name:", key="sheet_name")
sender_email = st.text_input("✉ Enter sender email:", key="sender_email")
email_password = st.text_input("🔑 Enter sender email password:", key="email_password", type="password")
recipient_email = st.text_input("📩 Enter recipient email:", key="recipient_email")

# Button to run analysis
if st.button("🚀 Run Analysis"):
    if uploaded_file and query and credentials_file and sheet_name and sender_email and email_password and recipient_email:
        
        # Save uploaded file locally
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run the main processing function
        st.write("🔄 Processing... Please wait.")
        
        result, analysis, sheet_url = process_and_query(file_path, query, credentials_file, sheet_name)
        
        # Display Results
        st.success("✅ Analysis Complete!")
        st.write("### Query Result:")
        st.write(result)
        st.write("### Google Sheets URL:")
        st.write(sheet_url)
        st.write("### Risk Analysis Results:")
        st.json(analysis)

        # Send Email
        subject = "📜 Legal and Risk Analysis Results"
        body = f"""
        Here are the results of the legal and risk analysis:

        Query Result: {result}

        Google Sheets URL: {sheet_url}

        Analysis Results: {json.dumps(analysis, indent=4)}
        """

        send_email(sender_email, email_password, recipient_email, subject, body)
        st.success("✅ Email Sent Successfully!")

    else:
        st.error("⚠ Please fill in all required fields.")

# =================== Backend Processing Functions =================== #
def load_and_preprocess(file_path):
    """Load and split a text document into 1000-character chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        document = file.read()

    chunks = [document[i:i+1000] for i in range(0, len(document), 1000)]
    return chunks

def risk_detection(chunks):
    """Analyze legal risks using an NLP model."""
    model_name = "google/flan-t5-base"
    nlp = pipeline("text2text-generation", model=model_name)

    results = []
    for chunk in chunks:
        prompt_analysis = f"Analyze the following text for legal risks:\n\n{chunk}"
        analysis_result = nlp(prompt_analysis, max_length=200, do_sample=False)
        analysis_text = analysis_result[0]['generated_text']

        prompt_recommendations = f"Provide recommendations to mitigate identified risks:\n\n{chunk}"
        recommendations_result = nlp(prompt_recommendations, max_length=200, do_sample=False)
        recommendations_text = recommendations_result[0]['generated_text']

        results.append({"context": chunk, "analysis": analysis_text, "recommendations": recommendations_text})
    return results

def export_to_sheets(data, sheet_name, credentials_file):
    """Export analysis results to Google Sheets."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(creds)

    spreadsheet = client.create(sheet_name)
    sheet = spreadsheet.sheet1

    df = pd.DataFrame(data)
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

    return spreadsheet.url

def process_and_query(file_path, query, credentials_file, sheet_name):
    """Run document analysis, query Watson AI, and save results."""
    chunks = load_and_preprocess(file_path)

    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    credentials = {
        "url": "https://eu-de.ml.cloud.ibm.com",
        "apikey": "ygkx3537EypWqZ6ziVsZe_2TWa52ha7nSiCdRJAfXMBu"
    }
    project_id = "4ec8c16d-4406-4dd2-92da-41d1718164ff"

    model_id = 'google/flan-ul2'
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 50,
        GenParams.MAX_NEW_TOKENS: 200,
        GenParams.TEMPERATURE: 0.5
    }
    model = Model(model_id=model_id, params=parameters, credentials=credentials, project_id=project_id)
    flan_ul2_llm = WatsonxLLM(model=model)

    qa = RetrievalQA.from_chain_type(
        llm=flan_ul2_llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=False
    )

    result = qa.invoke(query)
    analysis = risk_detection(chunks)

    output_path = "risk_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=4)

    sheet_url = export_to_sheets(analysis, sheet_name, credentials_file)
    return result, analysis, sheet_url

def send_email(email, password, recipient_email, subject, body):
    """Send an email with analysis results."""
    message = MIMEMultipart()
    message['From'] = email
    message['To'] = recipient_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(email, password)
            server.send_message(message)
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
