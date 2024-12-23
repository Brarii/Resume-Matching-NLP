# Resume Matcher - Flask Application

## Overview

This Flask-based web application helps match resumes with a given job description using Natural Language Processing (NLP) techniques. Users can upload resumes in `.pdf`, `.docx`, or `.txt` formats, input a job description, and get a ranking of the most relevant resumes based on similarity scores.

## Features

- **Upload Resumes**: Supports `.pdf`, `.docx`, and `.txt` file formats.
- **Job Description Input**: Users can input a job description directly in a text box.
- **Similarity Calculation**: Leverages TF-IDF vectorization and cosine similarity to evaluate resume-job description match.
- **User-Friendly Interface**: Designed with a simple HTML form for ease of use.

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn for TF-IDF Vectorization and Cosine Similarity
- **Document Processing**: PyPDF2, docx2txt
- **Frontend**: HTML, Bootstrap (for basic styling)

## Setup Instructions

### Prerequisites

- Python 3.x installed on your system.
- Necessary libraries:
  ```
  pip install Flask PyPDF2 docx2txt scikit-learn
 ```
 
