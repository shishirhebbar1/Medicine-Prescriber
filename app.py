import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from tensorflow.keras.models import load_model
import psycopg2
import requests
from tensorflow.keras.models import model_from_config
mapping = {
    'Colds & Flu': 'colds  flu',
    'Eczema': 'eczema',
    'Asthma': 'asthma',
    'Hypothyroidism': 'hypothyroidism',
    'Allergies': 'allergies',
    'Diabetes (Type 1)': 'diabetes type 1',
    'Diabetes (Type 2)': 'diabetes type 2',
    'Gastrointestinal': 'gastrointestinal',
    'Rheumatoid Arthritis': 'rheumatoid arthritis',
    'Depression': 'depression',
    'Cancer': 'cancer',
    'Stroke': 'stroke',
    'UTI': 'uti',
    'Fever': 'fever',
    'Hepatitis C': 'hepatitis c',
    'Migraine': 'migraine',
    'Unknown': 'unknown',
    'IBD (Bowel)': 'ibd bowel',
    'Bipolar Disorder': 'bipolar disorder',
    'Bronchitis': 'bronchitis',
    'Hypertension': 'hypertension',
    'Osteoporosis': 'osteoporosis',
    'Pneumonia': 'pneumonia',
    'Seizures': 'seizures',
    'Psoriasis': 'psoriasis',
    'COPD': 'copd',
    'Fibromyalgia': 'fibromyalgia',
    'Diarrhea': 'diarrhea',
    'Insomnia': 'insomnia',
    'AIDS/HIV': 'aidshiv',
    "Alzheimer's": 'alzheimers',
    'Schizophrenia': 'schizophrenia',
    'Gout': 'gout',
    'GERD (Heartburn)': 'gerd heartburn',
    'Urinary Incontinence': 'urinary incontinence',
    'Diabetes (Type 2)': 'diabetes type 2',
    'Menopause': 'menopause',
    'Swine Flu': 'swine flu',
    'Hair Loss': 'hair loss',
    'Acne': 'acne',
    'Hayfever': 'hayfever',
    'Erectile Dysfunction': 'erectile dysfunction',
    'Cholesterol': 'cholesterol',
    'ADHD': 'adhd',
    'Osteoarthritis': 'osteoarthritis',
    'Herpes': 'herpes',
    'Covid 19': 'covid 19',
    'Angina': 'angina',
    'Constipation': 'constipation',
    'Pain': 'pain',
    'Diabetes (Type 1)': 'diabetes type 1',
    'Obesity': 'obesity'
}

# Connection details for System1 (remote PostgreSQL server)
host = "192.168.1.27"  # IP address of System1
database = "DIC"        # Database name
user = "postgres"       # Username
password = "1234"  # Password

try:
    # Connect to the remote PostgreSQL server
    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cur = conn.cursor()

    # Execute a test query
    cur.execute("SELECT * FROM drug_treatments;")
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]  # Extract column names from the cursor

    # Create an empty DataFrame with the column names
    data = pd.DataFrame(columns=columns)

    # Add rows to the DataFrame
    for row in rows:
        data.loc[len(data)] = row

    # Print the resulting DataFrame
    print(data)


    # Close the cursor and connection
    cur.close()
    conn.close()

except Exception as e:
    print(f"Error: {e}")

# App title
st.title("Medicine Prescriber")

# Sidebar title
st.sidebar.title("Welcome!")

username = st.text_input("Enter the username:")

# Step 1: Ask for gender
gender = st.radio(
    "Select Gender:",
    options=["Male", "Female"],
    help="Choose your gender."
)

# Store the gender as a string
gender_selected = gender
st.write(f"### Gender Selected: {gender_selected}")

age = st.number_input(
    "Enter your age:", 
    min_value=7, 
    max_value=80, 
    value=25, 
    step=1,
    help="Please enter an age between 7 and 80."
)
fever = st.selectbox(
    "Do you have a fever?", 
    options=["Yes", "No"],
    help="Is fever present?"
)

Cough = st.selectbox(
    "Do you have cough?", 
    options=["Yes", "No"],
    help="Are you coughing more than usual?"
)

Fatigue = st.selectbox(
    "Do you have fatigue?", 
    options=["Yes", "No"],
    help="Is fatigue present?"
)

Difficulty_breathing = st.selectbox(
    "Do you have difficulty breathing?", 
    options=["Yes", "No"],
    help="Are you having trouble breathing?"
)

# Step 2: Show remaining dropdowns based on gender
# Dropdown for dosage_mg
dosage_mg = st.selectbox(
    "How Often Do You Consume Drugs:", 
    options=["Very often", "Sometimes", "Never"],
    help="Based on your drug history"
)

if dosage_mg == "Very often":
    drug_history = st.selectbox(
        "Do you have a history with drug abuse:", 
        options=["Yes", "No"],
        help="Have you abused drugs before?"
    )

# Dropdown for alcohol consumption
alcohol = st.selectbox(
    "Have you recently consumes any alcohol?:", 
    options=["Yes", "No"],
    help="Choose the level of alcohol consumption."
)

# Conditionally show the pregnancy status dropdown for females only
preg = None
if gender_selected == "Female":
    preg = st.selectbox(
        "Pregnancy Status:", 
        options=["Yes", "No"],
        help="Choose the pregnancy status."
    )

# List of symptoms
symptoms = [
    'Abdominal bloating', 'Abdominal cramps', 'Abdominal pain', 'Acne', 'Agitation', 'Back pain', 
    'Bad breath', 'Balance problems', 'Behavioral issues', 'Blackheads', 'Bladder problems', 
    'Bleeding', 'Bleeding mole', 'Blisters', 'Bloating', 'Blood in stool', 'Blood in urine', 
    'Bluish skin', 'Blurred vision', 'Blurry vision', 'Body aches', 'Bone fractures', 'Bone pain', 
    'Bone spurs', 'Bowel problems', 'Breast pain', 'Breast tenderness', 'Bullseye rash', 
    'Burning sensation when urinating', 'Cardiovascular problems', 'Change in breast shape', 
    'Change in existing mole', 'Changes in bowel habits', 'Chest discomfort', 'Chest pain', 
    'Chest tightness', 'Chills', 'Chronic cough', 'Chronic hunger', 'Cloudy urine', 
    'Cognitive issues', 'Concentration issues', 'Confusion', 'Congestion', 'Conjunctivitis', 
    'Constipation', 'Coughing', 'Coughing at night', 'Coughing up blood', 'Coughing up mucus', 
    'Cracked skin', 'Curved spine', 'Cysts', 'Daytime fatigue', 'Dehydration', 'Delayed development', 
    'Delayed growth', 'Delayed puberty', 'Delayed speech', 'Delusions', 'Depression', 
    'Depressive episodes', 'Developmental delays', 'Diarrhea', 'Diarrhea with blood', 
    'Difficulty breathing', 'Difficulty concentrating', 'Difficulty exhaling', 
    'Difficulty seeing at night', 'Difficulty sleeping', 'Difficulty swallowing', 
    'Difficulty urinating', 'Difficulty walking', 'Difficulty with communication', 
    'Difficulty with language', 'Difficulty with social interactions', 'Discharge from eyes', 
    'Discomfort when sitting', 'Disorganized thinking', 'Disorientation', 'Dizziness', 'Dry mouth', 
    'Dry patches', 'Dry skin', 'Dull ache in abdomen', 'Ear pain', 'Excess hair growth', 
    'Excess salivation', 'Excessive bleeding', 'Excessive worry', 'Eye pain', 'Facial pain', 
    'Fading colors', 'Fainting', 'Fatigue', 'Feelings of worthlessness', 'Fever', 'Flattened face', 
    'Fluid buildup in brain', 'Fluid drainage', 'Frequent infections', 'Frequent lung infections', 
    'Frequent nosebleeds', 'Frequent urination', 'Friendly personality', 'Gasping for air', 
    'Grating sensation', 'Hallucinations', 'Halos around lights', 'Headache', 'Headaches', 
    'Heart defects', 'Heart problems', 'Heat in the joint', 'Heavy bleeding', 'High fever', 
    'Hoarseness', 'Hunger', 'Impulsivity', 'Increased sweating', 'Increased thirst', 'Infertility', 
    'Insomnia', 'Intellectual disability', 'Irregular heartbeat', 'Irregular periods', 
    'Irritability', 'Itching', 'Itchy eyes', 'Itchy rash', 'Itchy skin', 'Jaundice', 'Jaw cramping', 
    'Joint pain', 'Lack of coordination', 'Lack of motivation', 'Large bruises', 
    'Learning difficulties', 'Limited range of motion', 'Long limbs', 'Loss of appetite', 
    'Loss of balance', 'Loss of consciousness', 'Loss of flexibility', 'Loss of height', 
    'Loss of interest', 'Loss of joint function', 'Loss of peripheral vision', 'Loud snoring', 
    'Low testosterone', 'Lower back pain', 'Lower right abdominal pain', 'Lump in breast', 
    'Lump in neck', 'Lump in testicle', 'Manic episodes', 'Memory loss', 'Mild headache', 
    'Mood changes', 'Mood swings', 'Morning headache', 'Muscle cramps', 'Muscle pain', 
    'Muscle spasms', 'Muscle stiffness', 'Muscle weakness', 'Nail pitting', 'Nasal congestion', 
    'Nausea', 'Nervousness', 'New mole', 'Night sweats', 'Nipple discharge', 'Nosebleeds', 
    'Numbness', 'Numbness in limbs', 'Oily skin', 'One hip higher than the other', 
    'Orthopedic problems', 'Pain behind eyes', 'Pain crises', 'Pain during intercourse', 
    'Pain in scrotum', 'Painful bowel movements', 'Painful mole', 'Painful periods', 
    'Painful urination', 'Pale skin', 'Palpitations', 'Paralysis', 'Pelvic pain', 
    'Persistent cough', 'Persistent sadness', 'Phlegm production', 'Pimples', 'Poor muscle tone', 
    'Poor weight gain', 'Postnasal drip', 'Rapid breathing', 'Rapid heart rate', 
    'Rapid heartbeat', 'Rash', 'Rectal bleeding', 'Red eyes', 'Red patches on skin', 
    'Redness', 'Reduced muscle mass', 'Repetitive behaviors', 'Restlessness', 'Runny nose', 
    'Salty skin', 'Seeing halos', 'Seizures', 'Sensitivity to light', 'Sensory sensitivities', 
    'Severe diarrhea', 'Severe headache', 'Severe joint pain', 'Severe upper abdominal pain', 
    'Shakiness', 'Short neck', 'Short stature', 'Shortness of breath', 'Silvery scales', 
    'Skin dimpling', 'Skin rash', 'Sleep problems', 'Slow healing of wounds', 'Slowed movement', 
    'Small ears', 'Sneezing', 'Social withdrawal', 'Sore throat', 'Speech difficulties', 
    'Staring spells', 'Stiffness', 'Stomach pain', 'Stooped posture', 'Strong-smelling urine', 
    'Sudden chest pain', 'Sudden headache', 'Sudden numbness', 'Sweating', 'Swelling', 
    'Swelling around anus', 'Swelling in hands and feet', 'Swelling in legs', 'Swollen abdomen', 
    'Swollen ankles', 'Swollen eyelids', 'Swollen glands', 'Swollen lymph nodes', 
    'Swollen salivary glands', 'Swollen tonsils', 'Tall stature', 'Tearing', 'Throat pain', 
    'Throbbing pain on one side', 'Tightness in chest', 'Tingling', 'Tiredness', 'Tooth pain', 
    'Tremors', 'Trouble hearing', 'Trouble speaking', 'Trouble walking', 
    'Trouble with coordination', 'Uneven shoulders', 'Unexplained weight loss', 
    'Upper abdominal pain', 'Upward slanting eyes', 'Urgency to defecate', 'Vision problems', 
    'Visual disturbances', 'Vomiting', 'Weak bones', 'Weakness', 'Weakness in legs', 
    'Weight gain', 'Weight loss', 'Wheezing', 'Widespread pain'
]

# Title
st.title("Choose the symptoms that you have")

# Dropdown for symptoms with multi-select
selected_symptoms = st.multiselect(
    "Select Symptoms",
    options=symptoms,
    help="Start typing to filter and select symptoms",
)
if dosage_mg == 'Very often':
    dosage_mg_mean = 30.78
    if drug_history == 'Yes':
        drug_history_mean = 4.47
    else:
        drug_history_mean = 4.59
elif dosage_mg == 'Sometimes':
    dosage_mg_mean = 226.51
    drug_history_mean = 4.59
else:
    dosage_mg_mean = 570.97
    drug_history_mean = 4.98

if alcohol == 'Yes':
    alcohol_mean = 0.49
else:
    alcohol_mean = 0.74

if preg:
    if preg == 'Yes':
        preg_mean = 2.29
    else:
        preg_mean = 2.63
else:
    preg_mean = 2.63

# Define the clean_text function
def clean_text(text):
    if isinstance(text, list):
        # Apply cleaning to each string in the list
        return [re.sub(r'[^\w\s]', '', t.lower()) for t in text]
    elif isinstance(text, str):
        # Clean a single string
        return re.sub(r'[^\w\s]', '', text.lower())
    else:
        raise ValueError("Input must be a string or a list of strings")

# Load the model
model = load_model('my_model.h5')
# Get the model configuration
config = model.get_config()

# Save configuration to a JSON file
with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=4)

# Load the modified configuration
with open('model_config.json', 'r') as f:
    updated_config = json.load(f)

# Recreate the model
model = model_from_config(updated_config)

# Save the updated model
model.save('updated_model.h5')
# Load additional preprocessing objects
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the prediction function
def predict_disease(model, tokenizer, label_encoder, scaler, new_text_input, new_binary_input):
    """Predicts the disease based on input text and binary features."""
    new_text_input = clean_text(new_text_input)  
    new_text_sequence = tokenizer.texts_to_sequences([new_text_input])
    new_text_padded = pad_sequences(new_text_sequence, maxlen=max_length, padding='post')
    
    # Scale the Age (last column of new_binary_input)
    new_binary_input[:, -1] = scaler.transform(new_binary_input[:, -1].reshape(-1, 1)).flatten()
    
    # Predict
    prediction = model.predict([new_text_padded, new_binary_input])
    predicted_class_index = np.argmax(prediction)
    predicted_disease = label_encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_disease

# Example Usage
new_text_input = selected_symptoms
def process_list(input_list):
    # Map "yes" to 1 and "no" to 0
    yes_no_map = {"yes": 1, "no": 0}
    
    # Map gender to 1 (male) and 0 (female)
    gender_map = {"male": 1, "female": 0}
    
    # Process the list
    processed_list = []
    for i, item in enumerate(input_list):
        if i < 4:  # First four elements: Fever, Cough, Fatigue, Breathing
            processed_list.append(yes_no_map.get(item.lower(), item))
        elif i == 4:  # Gender
            processed_list.append(gender_map.get(item.lower(), item))
        elif i == 5:  # Age
            processed_list.append(item)  # Retain age as is
    return processed_list

# Example usage
input_list = [fever, Cough, Fatigue, Difficulty_breathing, gender, age]
processed = process_list(input_list)
new_binary_input = np.array([processed])  # Binary features: [Fever, Cough, Fatigue, Breathing, Gender, Age]
max_length = 100  # Must match the maximum sequence length used during training
predicted_disease = predict_disease(model, tokenizer, label_encoder, scaler, new_text_input, new_binary_input)
st.write("Predicted Disease:", predicted_disease)
user_input = {
    "dosage_mg": dosage_mg_mean,
    "csa": drug_history_mean,
    "alcohol": alcohol_mean,
    "preg": preg_mean,
    "disease": mapping[predicted_disease]
}
# Function to insert a row into the symptoms table
def insert_row(predicted_disease, fever, Cough, Fatigue, Difficulty_breathing, age, gender, processed, username):
    username = username
    fever = fever
    cough = Cough
    fatigue = Fatigue
    difficulty_breathing = Difficulty_breathing
    age = age
    gender = gender
    extra_symptoms = processed
    mapped_disease = predicted_disease

    # Query to insert data and retrieve the new id
    insert_query = """
    INSERT INTO symptoms (id, username, fever, cough, fatigue, difficulty_breathing, age, gender, extra_symptoms, mapped_disease)
    VALUES ((SELECT COALESCE(MAX(id), 0) + 1 FROM symptoms), %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
    """

    data = (username, fever, cough, fatigue, difficulty_breathing, age, gender, extra_symptoms, mapped_disease)

    try:
        conn = psycopg2.connect(host=host, database=database, user=user, password=password)
        cur = conn.cursor()
        cur.execute(insert_query, data)
        new_id = cur.fetchone()[0]  # Fetch the newly inserted ID
        conn.commit()
        st.write(f"Row inserted successfully with ID: {new_id}, Username: {username}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error inserting row: {e}")
    return new_id

# Function to delete a row based on user-provided id and username
def delete_row(username):
    
    delete_query = "DELETE FROM symptoms WHERE username = %s;"
    try:
        conn = psycopg2.connect(host=host, database=database, user=user, password=password)
        cur = conn.cursor()
        cur.execute(delete_query, (username,))
        
        conn.commit()
        if cur.rowcount > 0:
            st.write(f"Username '{username}' deleted successfully.")
        else:
            st.write(f"No row found with Username '{username}'.")
        cur.close()
        conn.close()
    except Exception as e:
        st.write(f"Error deleting row: {e}")

if st.button("Insert Row into the database"):
    new_id = insert_row(predicted_disease, fever, Cough, Fatigue, Difficulty_breathing, age, gender, processed, username)

if st.button("Delete Row from the database"):
    delete_row(username)

if st.button("Get Recommendations"):
    response = requests.post("http://192.168.1.164:8000/recommend", json=user_input)
    if response.status_code == 200:
        recommendations = response.json().get("recommendations", [])
        df = pd.DataFrame(recommendations)

        # Display as a table in Streamlit
        st.write("### Drug Recommendations")
        st.table(df)
    else:
        st.error("Failed to fetch recommendations")
