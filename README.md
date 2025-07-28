# 🚗 CarDekho Price Prediction App

A machine learning web application that predicts car prices using XGBoost algorithm. Built with Streamlit for an interactive web interface.

## 🌟 Live Demo

**[Try the Live App →](https://your-app-name.streamlit.app)** _(Replace with your deployed URL)_

## 📊 Features

- **Real-time Price Prediction** using XGBoost ML model
- **Interactive Web Interface** with organized input sections
- **Price Categorization** (Budget/Mid-range/Premium/Luxury)
- **Multiple Format Display** (Rupees/Lakhs/Crores)
- **Model Performance Metrics** display
- **Responsive Design** for all devices

## 🤖 Model Performance

- **Algorithm:** XGBoost Regressor
- **R² Score:** 86.26% accuracy
- **Features:** 10 input features
- **Training Data:** 15,413 car records

## 🚀 Quick Start

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/noviciusss/CarPrice_Prediction.git
   cd CarPrice_Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser:** http://localhost:8501

### Using Docker

```bash
docker build -t cardekho-app .
docker run -p 8501:8501 cardekho-app
```

## 📱 How to Use

1. **Enter Car Details:**
   - Year of manufacture
   - Kilometers driven
   - Mileage (kmpl)
   - Engine displacement (CC)
   - Maximum power (bhp)
   - Number of seats
   - Model, seller type, fuel type, transmission

2. **Get Prediction:**
   - Click "Predict Price"
   - View price in multiple formats
   - See price category and insights

## 🔧 Input Features

| Feature | Description | Example |
|---------|-------------|---------|
| Year | Manufacturing year | 2018 |
| KM Driven | Total kilometers | 45,000 |
| Mileage | Fuel efficiency (kmpl) | 18.9 |
| Engine | Displacement (CC) | 1197 |
| Max Power | Power output (bhp) | 89.8 |
| Seats | Seating capacity | 5 |
| Model | Car model | Swift |
| Seller Type | Individual/Dealer | Individual |
| Fuel Type | Petrol/Diesel/CNG/LPG/Electric | Petrol |
| Transmission | Manual/Automatic | Manual |

## 📈 Sample Predictions

| Car Details | Predicted Price | Category |
|-------------|----------------|-----------|
| 2018 Swift, 45k km, Petrol, Manual | ₹5.12 Lakhs | Mid-range 💛 |
| 2020 City, 25k km, Petrol, Automatic | ₹8.45 Lakhs | Premium 🧡 |
| 2015 Alto, 80k km, Petrol, Manual | ₹2.85 Lakhs | Budget-friendly 💚 |

## 🏗️ Project Structure

```
cardekho-price-predictor/
├── app.py                    # Main Streamlit application
├── car_price_model.pkl       # Trained XGBoost model
├── cardekho_dataset.csv      # Training dataset
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── Procfile                 # Heroku deployment
├── setup.sh                 # Heroku setup script
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md                # Project documentation
```

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from your repository

### Heroku
```bash
heroku create your-app-name
git push heroku main
```

### Docker
```bash
docker build -t cardekho-app .
docker run -p 8501:8501 cardekho-app
```

For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 🛠️ Technologies Used

- **Python 3.9+**
- **Streamlit** - Web framework
- **XGBoost** - Machine learning model
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - ML preprocessing
- **Pickle** - Model serialization

## 📊 Model Details

### Training Process:
1. **Data Preprocessing:** 
   - Removed non-essential features (car_name, brand)
   - Applied log transformation to target variable
   - Used LabelEncoder for model names
   - StandardScaler for numerical features
   - OneHotEncoder for categorical features

2. **Model Training:**
   - Algorithm: XGBoost Regressor (base configuration)
   - Train-test split: 75%-25%
   - Random state: 42 for reproducibility

3. **Performance Metrics:**
   - R² Score: 86.26%
   - Mean Absolute Error: ₹102,236
   - Training on 15,413 records

## 🎯 Price Categories

- 💚 **Budget-friendly:** < ₹3 Lakhs
- 💛 **Mid-range:** ₹3-8 Lakhs
- 🧡 **Premium:** ₹8-15 Lakhs
- ❤️ **Luxury:** > ₹15 Lakhs

## 🔮 Future Enhancements

- [ ] Hyperparameter tuning for better accuracy
- [ ] Additional features (car condition, location)
- [ ] Model ensemble for improved predictions
- [ ] Real-time market data integration
- [ ] Mobile app version
- [ ] API endpoints for integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@noviciusss](https://github.com/noviciusss)
- LinkedIn: [yourprofile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Dataset source: CarDekho
- Streamlit team for the amazing framework
- XGBoost developers for the ML algorithm
- Open source community

## 📞 Support

If you have any questions or issues:
- 🐛 [Report bugs](https://github.com/noviciusss/CarPrice_Prediction/issues)
- 💡 [Request features](https://github.com/noviciusss/CarPrice_Prediction/issues)
- 📧 Contact: your.email@domain.com

---

**⭐ Star this repository if you found it helpful!**

![Car Price Prediction Demo](https://via.placeholder.com/800x400/1f77b4/ffffff?text=CarDekho+Price+Predictor+Demo)
