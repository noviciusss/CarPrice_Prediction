# 🚀 CarDekho Price Predictor - Deployment Guide

## 📋 Deployment Options

### 🌐 Option 1: Streamlit Cloud (Recommended)

**Steps:**
1. **Upload to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - CarDekho Price Predictor"
   git branch -M main
   git remote add origin https://github.com/yourusername/cardekho-price-predictor.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy"

3. **Required Files for Streamlit Cloud:**
   - ✅ `app.py` - Main application
   - ✅ `requirements.txt` - Dependencies
   - ✅ `car_price_model.pkl` - Trained model
   - ✅ `cardekho_dataset.csv` - Dataset
   - ✅ `.streamlit/config.toml` - Configuration

---

### 🔧 Option 2: Heroku Deployment

**Steps:**
1. **Install Heroku CLI:**
   - Download from [heroku.com](https://devcenter.heroku.com/articles/heroku-cli)

2. **Deploy:**
   ```bash
   heroku login
   heroku create cardekho-price-predictor
   git push heroku main
   heroku open
   ```

3. **Required Files for Heroku:**
   - ✅ `Procfile` - Process configuration
   - ✅ `setup.sh` - Setup script
   - ✅ `requirements.txt` - Dependencies
   - ✅ All app files

---

### 🐳 Option 3: Docker Deployment

**1. Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**2. Build and Run:**
```bash
docker build -t cardekho-app .
docker run -p 8501:8501 cardekho-app
```

---

### ☁️ Option 4: Railway Deployment

**Steps:**
1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Add environment variables if needed
4. Deploy automatically

---

### 🌍 Option 5: Render Deployment

**Steps:**
1. Go to [render.com](https://render.com)
2. Connect GitHub repository
3. Select "Web Service"
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 📁 Required Files Checklist

### ✅ Core Application Files:
- `app.py` - Main Streamlit application
- `car_price_model.pkl` - Trained XGBoost model
- `cardekho_dataset.csv` - Dataset (optional, has fallback)

### ✅ Deployment Configuration:
- `requirements.txt` - Python dependencies
- `Procfile` - Heroku process configuration
- `setup.sh` - Heroku setup script
- `.streamlit/config.toml` - Streamlit configuration
- `packages.txt` - System dependencies

### ✅ Documentation:
- `README.md` - Project documentation
- `DEPLOYMENT_GUIDE.md` - This file

---

## 🔧 Pre-deployment Checklist

### 1. Test Locally:
```bash
streamlit run app.py
```
Visit: http://localhost:8501

### 2. Verify Files:
- ✅ Model file exists and loads correctly
- ✅ Dataset file accessible (or fallback works)
- ✅ All dependencies in requirements.txt
- ✅ No hardcoded file paths

### 3. GitHub Repository Structure:
```
cardekho-price-predictor/
├── app.py
├── car_price_model.pkl
├── cardekho_dataset.csv
├── requirements.txt
├── Procfile
├── setup.sh
├── packages.txt
├── .streamlit/
│   └── config.toml
└── README.md
```

---

## 🎯 Recommended Deployment: Streamlit Cloud

**Why Streamlit Cloud?**
- ✅ Free hosting
- ✅ Easy GitHub integration
- ✅ Automatic updates
- ✅ Built for Streamlit apps
- ✅ Good performance
- ✅ Custom domains available

**Live App URL:** `https://your-app-name.streamlit.app`

---

## 🚨 Troubleshooting

### Common Issues:

1. **Model file too large:**
   - Use Git LFS for files > 100MB
   - Consider model compression

2. **Dependencies error:**
   - Ensure all packages in requirements.txt
   - Use exact versions that work

3. **Memory issues:**
   - Optimize model loading with @st.cache_resource
   - Reduce dataset size if needed

4. **File not found errors:**
   - Check file paths are relative
   - Ensure files are in repository

---

## 📊 Performance Optimization

### For Production:
1. **Caching:** Use @st.cache_data and @st.cache_resource
2. **Model Size:** Consider model compression
3. **Memory:** Monitor resource usage
4. **Error Handling:** Add try-catch blocks

---

## 🔐 Security Considerations

1. **No sensitive data in code**
2. **Use environment variables for configs**
3. **Validate user inputs**
4. **Add rate limiting if needed**

---

## 📈 Monitoring

### Track These Metrics:
- App uptime
- Response time
- User engagement
- Error rates
- Resource usage

---

**🎉 Your CarDekho Price Predictor is ready for deployment!**

Choose your preferred platform and follow the steps above. Streamlit Cloud is recommended for beginners.
