# Aspect-Based Sentiment Analysis for Service Industry

A full-stack Flask web application designed to analyze customer feedback and classify reviews as positive or negative for service-based businesses. The project also incorporates topic modeling to identify the most relevant themes in the input review text.

## Project Overview

This project helps businesses understand customer sentiment more deeply than a simple positive/negative label. Instead of only predicting the overall sentiment, it also extracts likely discussion topics from the review text, making it useful for service industries such as hospitality, telecom, banking, healthcare, and e-commerce support.

The system takes a customer message as input, predicts sentiment using a trained machine learning model, and shows the dominant topic or relevant keywords behind the review.

## Features

### User Features
- **Sentiment Analysis**: Analyze customer reviews and classify them as positive or negative
- **Topic Modeling**: Identify major themes and keywords in review text
- **Analytics Dashboard**: View sentiment analysis trends and historical data
- **User Authentication**: Secure sign-up and sign-in with OTP-based email verification
- **Bulk Upload**: Import and analyze multiple reviews at once with CSV support
- **Export Analytics**: Download analysis results in Excel format
- **Responsive Web Interface**: Modern, mobile-friendly HTML templates

### Admin Features
- **Admin Dashboard**: Access comprehensive admin control panel
- **User Management**: View, edit, and delete user accounts
- **Raw Data Viewer**: Inspect all analyzed reviews in the database
- **Analytics Export**: Export user analytics and metrics
- **Bulk Data Management**: Handle large-scale review imports and exports
- **Admin Analytics**: Monitor application usage and statistics

### Additional Features
- **Jupyter Notebook Integration**: Explore model and data analysis in notebooks
- **NLTK Data Auto-Download**: Automatic download of required NLP packages
- **Topic Modeling Logic**: Dedicated module for advanced topic extraction
- **Multiple Security Layers**: OTP verification, session management, admin authentication

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

## Application Routes & Endpoints

### Public Routes
- `/` or `/home` - Homepage
- `/about` - About page
- `/predict` - Sentiment prediction interface (GET/POST)
- `/analytics` - User analytics dashboard
- `/signup` - User registration page
- `/otp` - OTP verification (POST)
- `/signin` - User login page (GET/POST)
- `/logout` - User logout
- `/notebook` - Notebook viewer

### Admin Routes
- `/admin/login` - Admin login (GET/POST)
- `/admin` - Admin dashboard
- `/admin/analytics-export` - Export user analytics
- `/admin/export-analytics-csv` - Download analytics as CSV
- `/admin/edit/<user_id>` - Edit user details (GET/POST)
- `/admin/delete/<user_id>` - Delete user account (POST)
- `/admin/export-excel` - Export data to Excel
- `/admin/raw-data` - View all raw data and reviews

### Data Management Routes
- `/bulk_upload` - Bulk upload reviews (GET/POST)
- `/download_sample_template` - Download CSV template
- `/download_bulk_export` - Download bulk export data

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

## Configuration

Before running the application, set the following environment variables:

### Email Configuration (Optional - for OTP verification)
```bash
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### Application Secret (Optional)
```bash
SECRET_KEY=your_secret_key_here
```

If not set, the app uses default values:
- `SMTP_USER` and `SMTP_PASSWORD`: Email OTP functionality is disabled
- `SECRET_KEY`: Uses default `service-industry-admin-dashboard-secret`

## Run the Application

### Development Mode
From the project root, start the Flask app:

```bash
python app.py
```

Then open the browser and go to:

```text
http://127.0.0.1:5000/
```

### Production Deployment
The project includes `Procfile` and `runtime.txt` for Heroku deployment:

```bash
gunicorn app:app
```

## Database

The application stores user registration and analysis data in SQLite:

```text
signup.db
```

### Database Tables
- **users** - User account information and registration details
- **analyzed_reviews** - All sentiment analysis results with timestamps
- **admin_users** - Administrator account credentials and access control

This database is created automatically when the application starts and users interact with features that require persistence.

## Model Notes

The project uses pre-trained machine learning models:

- **model.pickle**: Vectorizer/preprocessing object for text transformation
- **model.sav**: Trained sentiment classification model (Joblib format)

These files are required for prediction to work correctly. To retrain the models with new data, use:

```bash
python retrain.py
```

## Important Notes

- **NLTK Data**: The app automatically downloads required NLTK data packages (stopwords, punkt, wordnet, omw-1.4) on first run
- **Email Setup**: Gmail users should use [App Passwords](https://support.google.com/accounts/answer/185833) instead of regular passwords for SMTP authentication
- **Database**: The SQLite database is created automatically; no manual setup required
- **Model Dependencies**: Ensure `model.sav` and `model.pickle` are in the project root
- **Python Version**: Requires Python 3.7 or higher
- **File Uploads**: Maximum file size for bulk upload can be configured in `app.py`
- **Session Security**: Session data is stored in memory; use persistent storage for production

## Development & Testing

### Running Notebooks
Open and run Jupyter notebooks for model exploration:

```bash
jupyter notebook Notebook.ipynb
```

### Sample Data
Test data is available in:
- `samples.csv` - Sample records for testing
- `data/train.csv` - Training dataset
- `data/test.csv` - Test dataset
- `data/sample_submission.csv` - Sample submission format

### Testing Bulk Upload
Use the `/bulk_upload` route and the provided CSV template (accessible via `/download_sample_template`)

## Deployment

The project is configured for Heroku deployment with:
- `Procfile` - Specifies web dyno with gunicorn
- `runtime.txt` - Python version specification
- Environment variables set in Heroku dashboard

For other cloud platforms (AWS, Azure, Google Cloud), modify the deployment configuration accordingly.

## Future Improvements

- [ ] Add advanced analytics dashboard with charts and visualizations
- [ ] Implement real-time sentiment tracking and alerts
- [ ] Improve aspect extraction for service-specific categories
- [ ] Add multi-class sentiment labels (neutral, mixed, very positive, very negative)
- [ ] Integrate with service industry APIs (hospitality, telecom, banking)
- [ ] Deploy to production with load balancing
- [ ] Add REST API endpoints for programmatic access
- [ ] Implement model versioning and A/B testing
- [ ] Add user feedback loop to improve model accuracy
- [ ] Create mobile app for on-the-go analysis
- [ ] Add multi-language support for sentiment analysis
- [ ] Implement data anonymization for GDPR compliance

## Dependencies

Core dependencies are managed in `requirements.txt`:
- Flask 3.1.3 - Web framework
- scikit-learn 1.9.0 - Machine learning library
- pandas 3.0.3 - Data manipulation
- numpy 2.5.1 - Numerical computing
- nltk 3.9.4 - Natural language processing
- gunicorn 26.0.0 - Production WSGI server
- joblib 1.5.2 - Model serialization

## License

This project is intended for educational and personal use. See LICENSE file for details.

## Support & Contributions

For issues, questions, or feature requests, please open an issue in the project repository.

## Author & Contact

**Project**: Aspect-Based Sentiment Analysis for Service Industry
**Purpose**: Educational demonstration of sentiment analysis and topic modeling
**Year**: 2024-2025

This project showcases full-stack web development skills with Flask, machine learning integration, and database management for the service industry domain.
