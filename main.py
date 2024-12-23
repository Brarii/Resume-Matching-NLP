import os
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)

# Set upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt'}

# Load the dataset
data = pd.read_csv('data/dataset.csv')

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to extract text from PDF
def extract_text_from_pdf(filepath):
    from PyPDF2 import PdfReader
    with open(filepath, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(filepath):
    from docx import Document
    doc = Document(filepath)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

# Preprocess dataset 
def preprocess_data():
    # Remove rows with missing values
    cleaned_data = data.dropna(subset=['Resume', 'Category'])

    # Remove duplicates
    cleaned_data = cleaned_data.drop_duplicates(subset=['Resume'])

    # Normalize text
    cleaned_data['Resume'] = cleaned_data['Resume'].str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()

    # Extract resumes and categories
    resumes = cleaned_data['Resume'].values
    categories = cleaned_data['Category'].values

    return resumes, categories

# Train the model
def train_model():
    print("Training the model...")
    resumes, categories = preprocess_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(resumes, categories, test_size=0.2, random_state=42)

    # Initialize the vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Transform the resume text
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the classifier 
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Save the model and vectorizer to files
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

    print("Training complete. Model saved.")

    # Evaluate the model on the test set
    accuracy = model.score(X_test_tfidf, y_test)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# Load the pre-trained model and vectorizer
def load_model():
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    return model, vectorizer

# Route to render the page and process the form submission
@app.route("/")
def matchresume():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['GET', 'POST'])
def matcher():
    if request.method == 'POST':
        # Retrieve job description and uploaded resumes
        job_description = request.form['job_description']
        uploaded_files = request.files.getlist('resumes')

        # Create a list to store resume contents
        resumes_content = []

        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                try:
                    # Handle different file types
                    if filename.lower().endswith('.txt'):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            resumes_content.append(f.read())            
                    elif filename.lower().endswith('.pdf'):
                        resumes_content.append(extract_text_from_pdf(filepath))
                    elif filename.lower().endswith('.docx'):
                        resumes_content.append(extract_text_from_docx(filepath))
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    resumes_content.append("")  # Add empty string if there's an error

        # Combine job description with resumes for similarity calculation
        all_texts = [job_description] + resumes_content

        # Load the pre-trained model and vectorizer
        model, vectorizer = load_model()

        # Convert the job description and resumes into the same TF-IDF vector space
        all_texts_tfidf = vectorizer.transform(all_texts)

        # Get the prediction scores for each resume
        job_desc_vector = all_texts_tfidf[0]
        resume_vectors = all_texts_tfidf[1:]

        # Get the category prediction for each resume using the model
        predicted_categories = model.predict(resume_vectors)

        # Calculate similarity scores using cosine similarity
        cosine_sim = cosine_similarity(job_desc_vector, resume_vectors).flatten()

        # Sort resumes based on similarity scores
        sorted_indices = cosine_sim.argsort()[::-1]
        top_resumes = [uploaded_files[i].filename for i in sorted_indices]
        top_similarity_scores = [cosine_sim[i] for i in sorted_indices]
        top_categories = [predicted_categories[i] for i in sorted_indices]

        # Render the template with the results
        return render_template('matchresume.html', message="Matching Completed", 
                               top_resumes=top_resumes, similarity_scores=top_similarity_scores,
                               matched_categories=top_categories)

    return render_template('matchresume.html')


if __name__ == '__main__':
    if not os.path.exists('model.joblib') or not os.path.exists('vectorizer.joblib'):
        train_model()  
    app.run(debug=True)
