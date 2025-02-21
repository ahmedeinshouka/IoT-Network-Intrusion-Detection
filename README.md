# 🚀 Advanced Network Traffic Analyzer



---

## 🌟 Overview

Unleash the power of machine learning to safeguard your networks with the **Advanced Network Traffic Analyzer**—a cutting-edge, web-based tool designed to detect, analyze, and visualize network traffic anomalies with precision. Built on Flask, Pandas, and Plotly, this application empowers network security professionals and researchers to upload CSV files or manually input network traffic features, predict patterns (e.g., Normal, DoS, MITM ARP Spoofing, Mirai attacks, Scan), and explore interactive, data-driven insights through stunning visualizations.

This project harnesses state-of-the-art random forest classifiers to deliver high-confidence predictions, making it an indispensable asset for securing IoT network intrusion datasets. Dive into a seamless, user-friendly interface that transforms raw data into actionable intelligence!

---

## 🔥 Key Features

- **👤 File Upload**: Effortlessly upload CSV files containing network traffic data for instant analysis.
- **✍️ Manual Input**: Enter traffic features manually for real-time, on-the-fly predictions.
- **🧠 Machine Learning Predictions**: Leverage pre-trained random forest models to predict traffic labels, categories, and subcategories with confidence scores (e.g., Anomaly/Normal, DoS/Mirai/Normal/Scan, specific attack types).
- **📊 Interactive Visualizations**: Explore dynamic charts (pie charts, histograms, heatmaps, bar charts) powered by Plotly to visualize predictions, feature distributions, correlations, and trends.
- **⏳ Progress Tracking**: A sleek progress bar keeps you informed as results are rendered.
- **💾 Export Results**: Download analysis results as a CSV file for further processing or reporting.

---

## 🛠 Prerequisites

Before launching this powerhouse, ensure you have the following installed:

- **Python** (3.8 or higher)
- **Flask** (web framework)
- **Pandas** (data manipulation)
- **NumPy** (numerical operations)
- **Plotly** (data visualization)
- **scikit-learn** (for machine learning models)
- **joblib** (for loading pre-trained models)
- **Flask-CORS** (for handling cross-origin requests)

Install the dependencies with a single command:

```bash
pip install flask pandas numpy plotly scikit-learn joblib flask-cors
```

---

## 🚀 Installation

### Clone the Repository:

Clone this repo to your local machine:

```bash
git clone https://github.com/AhmedEinshouka/IoT-NETWORK-INTRUSION-DATA.git
cd IoT-NETWORK-INTRUSION-DATA/API
```

### Set Up the Environment:

Create a virtual environment (recommended for a clean setup):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, generate one:

```bash
pip freeze > requirements.txt
```

### Prepare the Project:

- Ensure directories like `static/`, `templates/`, `Models/`, `uploads/`, `results/`, `src/`, and `validation_data/` are present with their respective files (e.g., `logo.png`, `index.html`, pre-trained models, datasets).
- Verify `logo.png` is in `API/static/` for the web interface.

### Configure Environment Variables:

Optionally, set a secure `SECRET_KEY` for Flask:

```bash
export SECRET_KEY="your-secret-key-here"
```

---

## ⚡ Usage

### Run the Application:

Navigate to `API/` and start the Flask app:

```bash
python app.py
```

Access it at `http://localhost:5000/` in your browser.

### Explore the Interface:

- **Upload Data**: Use the "File Upload" form to analyze CSV files with network traffic data.
- **Manual Analysis**: Input traffic features (e.g., Source Port, Protocol) manually and click "Analyze Data" for instant insights.
- **View Results**: See predictions, statistics, and interactive visualizations like confidence distributions, feature correlations, and trends.
- **Export Findings**: Download results as a CSV file for deeper analysis or reporting.

---

## 📦 Project Structure

```text
IoT-NETWORK-INTRUSION-DATA/
├── API/
│   ├── app.py              # Flask backend logic
│   ├── app.log            # Application logs
│   ├── Dockerfile         # Docker setup (optional)
│   ├── static/            # Static assets (e.g., logo.png)
│   │   └── logo.png       # Project logo
│   ├── Models/            # Pre-trained ML models
│   │   ├── Cat_Models/    # Category prediction models
│   │   ├── Label_Models/  # Label prediction models
│   │   └── Sub_Cat_Models/ # Subcategory prediction models
│   ├── results/           # Analysis output storage
│   ├── src/               # Source datasets
│   │   └── IoT Network Intrusion Dataset.csv
│   ├── templates/         # HTML templates
│   │   └── index.html     # Web interface
│   ├── uploads/           # Uploaded CSV files
│   └── validation_data/   # Validation datasets
│       ├── category_validation.csv
│       ├── label_validation.csv
│       ├── subcategory_validation.csv
```

---

## 🤝 Contributing

Join us in enhancing network security! Here’s how to contribute:

### Fork the Repo:
Create your own copy on GitHub.

### Create a Feature Branch:

```bash
git checkout -b feature/your-feature-name
```

### Make Changes:
Implement your improvements, following PEP 8 style guidelines, and add tests or documentation as needed.

### Commit Your Changes:

```bash
git commit -m "Add feature: Description of your changes"
```

### Push to GitHub:

```bash
git push origin feature/your-feature-name
```

### Submit a Pull Request:
Open a PR with a clear description of your changes, and we’ll review it!

We value all contributions—whether code, documentation, or ideas. Let’s build a stronger tool together! 🌍

---

## 🌟 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

🚀 Use it freely, modify it, and share your innovations!

---

## 🙏 Acknowledgments

- **Open-Source Community**: A huge thanks to the creators of Flask, Pandas, Plotly, scikit-learn, and joblib for their incredible tools.
- **Data Contributors**: Gratitude to the developers of the IoT Network Intrusion Dataset, enabling robust model training and testing.
- **Inspiration**: Inspired by the need for advanced network security solutions in the IoT era.

