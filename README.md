# Diabetes Health Predictor – AI Doctor Portal

A beautiful, production-ready hospital-style web application for diabetes risk prediction using Machine Learning and AI-powered medical report generation.

## 🏥 Features

- **Patient Registration System** - Comprehensive patient information capture
- **ML-Powered Prediction** - Accurate diabetes risk assessment using trained model
- **AI Doctor Reports** - Professional medical reports generated using Groq LLM
- **Modern Hospital UI** - Clean, responsive design with medical theme
- **Real-time Results** - Instant prediction with confidence scores
- **Downloadable Reports** - Save AI-generated reports as text files

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Groq API key ([Get one here](https://console.groq.com/))

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd "c:\Users\navee\Cisco Packet Tracer 8.2.2\saves\certificates\Diabetics-Agent"
   ```

2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   - Copy `.env.example` to `.env`
   - Add your Groq API key (and any Firebase overrides if needed):
   ```env
   GROQ_API_KEY=your_actual_api_key_here
   ```

5. **Run the Flask application locally**
   ```bash
   python flask_app.py
   ```

6. **Open your browser**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
Diabetics-Agent/
│
├── Dockerfile                # Container definition for Cloud Run
├── .dockerignore             # Build context exclusions
├── flask_app.py              # Main Flask application
├── auth.py                   # Authentication helpers
├── firebase_config.py        # Firebase integration layer
├── requirements.txt          # Production Python dependencies
├── .env.example              # Template for environment variables
├── README.md                 # Project documentation
│
├── templates/               # Jinja templates (landing, dashboard, reports, etc.)
│
├── static/                  # CSS and JavaScript assets
│
├── artifacts/               # ML model artifacts
│   ├── model.pkl
│   ├── scaler.pkl
│   └── proprocessor.pkl
│
├── reports/                 # Generated AI reports (auto-created)
│
├── src/                     # Source code modules
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   └── utils.py
│
├── data/
│   └── raw/
│       └── diabetes.csv     # Training dataset snapshot
│
├── firebase-service-account.template.json  # Sample service account layout
├── retrain_model.py         # Offline model training script
└── artifacts/model_info.txt # Model metadata

```

## 🐳 Run with Docker

Build the production image and run it locally:

```powershell
docker build -t diabetes-health-predictor .
docker run -p 8080:8080 --env GROQ_API_KEY=your_actual_api_key diabetes-health-predictor
```

The container listens on port `8080` by default (Cloud Run requirement). Override additional environment variables with `--env` flags as needed.

## ☁️ Deploy to Google Cloud Run

1. **Authenticate and choose your project**
   ```powershell
   gcloud auth login
   gcloud config set project YOUR_GCP_PROJECT_ID
   ```
2. **Build and push the container image with Cloud Build**
   ```powershell
   gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/diabetes-health-predictor
   ```
3. **Deploy to Cloud Run (fully managed)**
   ```powershell
   gcloud run deploy diabetes-health-predictor \ 
     --image gcr.io/YOUR_GCP_PROJECT_ID/diabetes-health-predictor \ 
     --platform managed \ 
     --region YOUR_REGION \ 
     --allow-unauthenticated \ 
     --set-env-vars GROQ_API_KEY=your_actual_api_key
   ```
4. **Optional: manage secrets securely**
   - Store `GROQ_API_KEY` (and any Firebase credentials) in **Secret Manager**.
   - Replace `--set-env-vars` with `--set-secrets GROQ_API_KEY=projects/.../secrets/...:latest` for runtime secret injection.
   - If you need Firebase Admin SDK, upload `firebase-service-account.json` to Secret Manager and mount it via Cloud Run volume.
5. **Verify deployment** – Cloud Run outputs a service URL; visit it and log in with your test account.

> ℹ️ Cloud Run automatically handles scaling, HTTPS certificates, and log aggregation. Remember to restrict access and rotate API keys in production.

## 🎯 How to Use

1. **Register Patient**
   - Fill in patient details (Name, Age, Sex, Contact, Address)

2. **Enter Medical Test Results**
   - Input all required medical parameters:
     - Glucose Level (mg/dL)
     - Blood Pressure (mmHg)
     - BMI
     - Diabetes Pedigree Function
     - And other optional parameters

3. **Get Prediction**
   - Click "Predict Risk" button
   - View instant results with confidence score

4. **Generate AI Report**
   - Click "Generate Doctor Report"
   - AI creates a comprehensive medical diagnosis
   - Download the report for records

## 🔧 API Endpoints

- `GET /` - Home page with forms
- `POST /predict` - ML prediction endpoint
- `POST /report` - AI report generation
- `GET /download_report/<filename>` - Download report file
- `GET /health` - Health check endpoint

## 🎨 Features Highlight

### Medical Test Parameters
- **Pregnancies** - Number of pregnancies
- **Glucose** - Plasma glucose concentration
- **Blood Pressure** - Diastolic blood pressure (mm Hg)
- **Skin Thickness** - Triceps skin fold thickness (mm)
- **Insulin** - 2-Hour serum insulin (mu U/ml)
- **BMI** - Body mass index (weight in kg/(height in m)²)
- **Diabetes Pedigree Function** - Diabetes genetic predisposition
- **Age** - Patient age in years

### Design Philosophy
- 🎨 Modern medical theme with soft blue and teal colors
- 📱 Fully responsive for mobile and desktop
- ⚡ Smooth animations and transitions
- 🔒 Form validation and error handling
- ♿ Accessible and user-friendly interface

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **ML**: scikit-learn, NumPy, Pandas
- **AI**: Groq LLM (via LangChain)
- **Fonts**: Google Fonts (Poppins)
- **Icons**: Font Awesome 6

## 📊 Model Information

The ML model is trained on the Pima Indians Diabetes Database and predicts diabetes risk based on diagnostic measurements.

## 🔐 Security Notes

- Never commit your `.env` file to version control
- Keep your Groq API key secure
- Use environment variables for sensitive data
- Implement rate limiting for production deployment

## 🚀 Production Deployment

For production deployment:

1. Set `debug=False` in `flask_app.py`
2. Use a production WSGI server (Gunicorn, uWSGI)
3. Set up proper error logging
4. Implement authentication if needed
5. Use HTTPS for secure communication
6. Set up CORS policies appropriately

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 flask_app:app
```

## 🤝 Contributing

This is a medical diagnostic tool. Ensure all contributions maintain:
- Medical accuracy
- Patient privacy
- Professional standards
- Code quality

## ⚠️ Disclaimer

This application is for educational and screening purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## 📝 License

© 2025 Diabetes Health Predictor - AI Doctor Portal

## 👨‍⚕️ About

**Dr. Ramesh Kumar Hospital**  
AI-Powered Medical Diagnostic System

---

**Built with ❤️ for better healthcare**
