# Aspect-Based Sentiment Analysis for Service Industry

A full-stack Flask web application designed to analyze customer feedback and classify reviews as positive or negative for service-based businesses. The project also incorporates topic modeling to identify the most relevant themes in the input review text.

## Project Overview

This project helps businesses understand customer sentiment more deeply than a simple positive/negative label. Instead of only predicting the overall sentiment, it also extracts likely discussion topics from the review text, making it useful for service industries such as hospitality, telecom, banking, healthcare, and e-commerce support.

The system takes a customer message as input, predicts sentiment using a trained machine learning model, and shows the dominant topic or relevant keywords behind the review.

## Features

- Sentiment analysis for customer reviews
- Positive/negative classification using a trained model
- Topic modeling to identify major themes in the text
- Flask-based web interface
- Sign-in and sign-up flow with OTP-based registration
- Responsive HTML templates and modern UI
- Notebook support for model exploration and analysis

## Tech Stack

- Python
- Flask
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Joblib
- SQLite
- HTML/CSS/JS

## Project Structure

```text
Aspect-Based Sentiment Analysis for Service Industry/
├── app.py                     # Main Flask application
├── topic_modelling.py         # Topic modeling logic
├── model.sav                  # Trained sentiment model
├── model.pickle              # Vectorizer / preprocessing object
├── retrain.py                # Retraining script (if used)
├── requirements.txt          # Python dependencies
├── Procfile                  # Heroku deployment config
├── runtime.txt               # Runtime version info
├── samples.csv               # Sample records
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── static/                   # Static CSS, JS, images, and assets
├── templates/                # HTML pages for the web app
├── Notebook.ipynb            # Jupyter notebook for experiments
├── Notebook-Copy1.ipynb      # Backup notebook copy
├── signup.db                 # SQLite database created during signup
└── README.md                 # Project documentation
```

## How the App Works

1. A user enters a review or message in the web form.
2. The input is converted into a numeric vector using a saved preprocessing object.
3. A sentiment model predicts whether the review is positive or negative.
4. The system also applies topic modeling to extract the main discussion theme from the text.
5. The result is displayed in the browser.

## Installation

1. Clone or download the project.
2. Open a terminal in the project folder.
3. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

From the project root, start the Flask app:

```bash
python app.py
```

Then open the browser and go to:

```text
http://127.0.0.1:5000/
```

## Environment Variables

The signup route uses email OTP verification. For this to work properly, set the following environment variables before running the app:

```bash
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

> If these values are not configured, the app will show an error message when the user tries to sign up.

## Database

The application stores user registration information in SQLite using the file:

```text
signup.db
```

This database is created automatically when a user completes the sign-up and OTP verification flow.

## Model Notes

The project uses:

- a saved vectorizer object: `model.pickle`
- a trained ML model: `model.sav`

These files are required for prediction to work correctly.

## Important Notes

- The app depends on NLTK data packages such as stopwords, punkt, wordnet, and omw-1.4. These are downloaded automatically during runtime if missing.
- The project is built for demonstration and educational use and can be extended for production deployment.
- For deployment, additional configuration and security improvements may be needed.

## Future Improvements

- Add a proper dashboard for analytics
- Improve aspect extraction for service categories
- Add multi-class sentiment labels such as neutral, mixed, and very positive
- Use a larger and cleaner service-industry dataset
- Deploy to Heroku, Render, or Azure

## License

This project is intended for educational and personal use unless otherwise specified by the owner.

## Author

This project was developed as an Aspect-Based Sentiment Analysis application for the service industry.
