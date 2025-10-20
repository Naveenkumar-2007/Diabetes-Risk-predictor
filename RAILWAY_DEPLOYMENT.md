# Railway Deployment Guide

## Prerequisites
✅ Code pushed to GitHub: `https://github.com/Naveenkumar-2007/Diabetes-Risk-predictor`
✅ Clean repository (unnecessary files removed)
✅ All MLOps components implemented and tested

## Quick Deploy Steps

### 1. Create Railway Project
1. Go to [Railway.app](https://railway.app)
2. Click **"Start a New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose: `Naveenkumar-2007/Diabetes-Risk-predictor`

### 2. Configure Environment Variables
Add these in Railway dashboard → Variables:

**Required Variables:**
```bash
# Flask Configuration
FLASK_APP=flask_app.py
FLASK_ENV=production
SECRET_KEY=<generate-random-secret-key>

# Firebase Credentials (from firebase-service-account.json)
FIREBASE_TYPE=service_account
FIREBASE_PROJECT_ID=<your-project-id>
FIREBASE_PRIVATE_KEY_ID=<your-key-id>
FIREBASE_PRIVATE_KEY="<your-private-key>"  # Include quotes
FIREBASE_CLIENT_EMAIL=<your-service-account-email>
FIREBASE_CLIENT_ID=<your-client-id>
FIREBASE_AUTH_URI=https://accounts.google.com/o/oauth2/auth
FIREBASE_TOKEN_URI=https://oauth2.googleapis.com/token
FIREBASE_AUTH_PROVIDER_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
FIREBASE_CLIENT_CERT_URL=<your-cert-url>
FIREBASE_UNIVERSE_DOMAIN=googleapis.com

# Groq API (for AI analysis)
GROQ_API_KEY=<your-groq-api-key>

# Email Configuration (optional, for forgot password)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_ADDRESS=<your-email@gmail.com>
EMAIL_PASSWORD=<your-app-password>
```

**Optional MLOps Variables:**
```bash
# MLflow (defaults work fine, but can customize)
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=diabetes-prediction
```

### 3. Railway Settings
- **Start Command**: `gunicorn flask_app:app --bind 0.0.0.0:$PORT`
- **Build Command**: (Leave empty - Railway auto-detects)
- **Root Directory**: `/`

### 4. Domain Setup (After Deployment)
1. Railway will auto-generate a domain like: `your-app.up.railway.app`
2. **Optional**: Add custom domain in Railway → Settings → Domains

## Important Notes

### Files NOT Pushed to Git (Excluded by .gitignore)
These are generated at runtime and don't need to be in the repository:
- `mlflow.db` - MLflow tracking database (regenerated on deploy)
- `mlruns/` - MLflow experiment runs
- `model_registry/` - MLflow model registry
- `logs/predictions/*.jsonl` - Prediction logs
- `logs/drift_reports/*.json` - Drift detection reports
- `firebase-service-account.json` - Sensitive credentials (use env vars instead)
- `venv/` - Virtual environment
- `__pycache__/` - Python cache
- `.pytest_cache/` - Test cache

### What IS Included in Deployment
✅ Source code (`flask_app.py`, `mlops/`, `src/`, etc.)
✅ Trained model artifacts (`artifacts/model.pkl`, `artifacts/scaler.pkl`)
✅ Model metadata (`artifacts/model_metadata.json`)
✅ Reference distribution (`data/processed/reference_distribution.pkl`)
✅ Dependencies (`requirements.txt`)
✅ Templates and static files (`templates/`, `static/`)
✅ Configuration (`mlops_config.py`)

## Testing Deployment

### 1. Check Health Endpoints
```bash
# Application health
curl https://your-app.up.railway.app/

# MLOps API health
curl https://your-app.up.railway.app/mlops/api/health
```

### 2. Test MLOps Endpoints
```bash
# Get model info
curl https://your-app.up.railway.app/mlops/api/model/info

# Check monitoring stats
curl https://your-app.up.railway.app/mlops/api/stats/summary

# Check drift detection
curl https://your-app.up.railway.app/mlops/api/monitoring/drift
```

### 3. Test Prediction
```bash
curl -X POST https://your-app.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

## Monitoring After Deployment

### View Logs in Railway
- Go to Railway dashboard
- Click on your project
- View **"Deployments"** tab
- Click on latest deployment → **"View Logs"**

### MLOps Monitoring
- Access monitoring endpoints via `/mlops/api/*`
- Predictions are automatically logged
- Drift detection runs on demand
- Reference distribution already saved in repo

## Troubleshooting

### Issue: Import Errors
**Solution**: Ensure all dependencies in `requirements.txt` are compatible
```bash
# Railway installs automatically, but you can test locally:
pip install -r requirements.txt
```

### Issue: Firebase Authentication Fails
**Solution**: 
1. Verify all Firebase env vars are set correctly
2. Check `FIREBASE_PRIVATE_KEY` includes quotes and `\n` for newlines
3. Ensure service account has correct permissions in Firebase console

### Issue: Model Not Found
**Solution**: Verify these files exist in repository:
- `artifacts/model.pkl`
- `artifacts/scaler.pkl`
- `artifacts/model_metadata.json`

### Issue: MLflow Database Error
**Solution**: Railway uses ephemeral storage
- MLflow tracking database recreates on each deploy (this is OK)
- Model artifacts are saved separately in `artifacts/`
- For persistent MLflow, use external database (optional)

## Performance Optimization

### 1. Reduce Slug Size
Already done! We removed:
- Old duplicate MLOps files
- Test files
- Development artifacts
- Runtime logs

### 2. Cold Start Optimization
Railway may have cold starts. To minimize:
- Keep instance awake with uptime monitoring (UptimeRobot, etc.)
- Enable Railway's "Always On" (paid feature)

### 3. Database Recommendations
For production scale:
- **Firebase**: Already configured (good for user data)
- **MLflow**: Currently SQLite (works for small-medium scale)
- **Upgrade path**: Use PostgreSQL for MLflow if needed

## Cost Estimate (Railway)

**Free Tier:**
- $5 credit/month
- Good for testing and small projects
- May sleep after inactivity

**Hobby Plan ($5/month):**
- Always-on deployments
- Better for production use
- 500+ monthly active users

## Next Steps After Deployment

1. ✅ **Test all features** (login, prediction, reports, admin dashboard)
2. ✅ **Monitor logs** for any errors
3. ✅ **Test Google OAuth** (add Railway domain to Firebase authorized domains)
4. ✅ **Test email notifications** (forgot password flow)
5. ✅ **Verify MLOps endpoints** work correctly
6. ✅ **Set up monitoring alerts** (optional)

## Railway Firebase Setup

### Add Railway Domain to Firebase
1. Go to Firebase Console → Authentication → Settings
2. Under **"Authorized domains"**, add:
   - `your-app.up.railway.app`
   - Any custom domain

### Update OAuth Redirect URIs
1. Google Cloud Console → APIs & Services → Credentials
2. Edit OAuth 2.0 Client
3. Add authorized redirect URIs:
   - `https://your-app.up.railway.app/callback`

## Success Checklist

- [ ] Code pushed to GitHub
- [ ] Railway project created
- [ ] Environment variables configured
- [ ] Deployment successful
- [ ] Health check passes
- [ ] Login/register works
- [ ] Google OAuth works
- [ ] Predictions work
- [ ] MLOps endpoints accessible
- [ ] Reports generate correctly
- [ ] Admin dashboard accessible

---

## Support

If you encounter issues:
1. Check Railway logs first
2. Verify environment variables
3. Test endpoints with curl
4. Check Firebase console for auth issues

**Your MLOps system is production-ready! 🚀**

All 25 tests passing | Model: XGBoost 75.97% accuracy | Monitoring: Active
