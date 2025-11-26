# Smart-Weather-Forecasting-System


A machine learning-based weather forecasting application designed to analyze climate data and predict weather conditions to assist in agricultural decision-making.

## 📄 Abstract
[cite_start]Weather prediction is crucial for the agricultural sector to minimize losses by implementing necessary preventative actions[cite: 52, 53]. [cite_start]This project utilizes machine learning algorithms to effectively identify and forecast weather parameters[cite: 54]. [cite_start]The system processes historical weather data to predict future conditions, utilizing a web application interface for easy accessibility[cite: 58].

## ✨ Key Features
* [cite_start]**User Authentication:** Secure Sign-Up and Login functionality using CSV-based storage[cite: 298, 304].
* [cite_start]**Data Preprocessing:** Includes handling missing values (replacing nulls with 0) and label encoding to convert string data into numeric values[cite: 119, 120].
* [cite_start]**Weather Prediction:** Users can input specific parameters (Date, Temperature, Precipitation, Wind Speed) to generate a weather forecast (e.g., Sun, Rain)[cite: 347].
* **Visualization:** Graphical representation of data including:
    * [cite_start]Temperature over time[cite: 347].
    * [cite_start]Monthly precipitation levels[cite: 347].
    * [cite_start]Correlation heatmaps and pair plots[cite: 347].
    * [cite_start]Weather type distribution[cite: 347].
* [cite_start]**Performance Metrics:** Displays experimental results including Accuracy, Precision, Recall, F1-score, and Confusion Matrix[cite: 56].

## 🛠️ Technology Stack
* [cite_start]**Language:** Python[cite: 166].
* [cite_start]**Web Framework:** Streamlit[cite: 58, 285].
* [cite_start]**Data Processing:** Pandas, CSV[cite: 113, 286].
* **Machine Learning:**
    * [cite_start]Logistic Regression[cite: 55].
    * [cite_start]K-Nearest Neighbors (KNN)[cite: 55].
    * [cite_start]XGBoost (mentioned in system diagrams)[cite: 212].
    * [cite_start]Random Forest (mentioned in application screenshots)[cite: 347].
* [cite_start]**IDE:** Anaconda Navigator – Spyder / VS Code[cite: 167].

## ⚙️ System Architecture
The system follows a structured pipeline:
1.  [cite_start]**Input Data:** Collection of weather datasets (Visibility, Wind Speed, Temperature, Humidity, etc.)[cite: 112].
2.  [cite_start]**Preprocessing:** Cleaning data and removing irrelevant/corrupted entries[cite: 118].
3.  [cite_start]**Data Splitting:** Dividng data into 80% Training and 20% Testing sets[cite: 123].
4.  [cite_start]**Classification:** Applying ML algorithms to detect weather patterns[cite: 135].
5.  [cite_start]**Result Generation:** Outputting predictions and performance analysis[cite: 140].

[cite_start]*(Refer to Figure 6.1 in the project documentation [cite: 211])*
# Basic Weather App
## 🚀 Installation & Usage

### Prerequisites
* [cite_start]Python 3.x installed[cite: 166].
* Required libraries: `streamlit`, `pandas`, `sklearn`, `matplotlib`, `seaborn`.

### Steps
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/venkatesh01-t/Smart-Weather-Forecasting-System.git](https://github.com/venkatesh01-t/Smart-Weather-Forecasting-System.git)
    cd basic-weather-app
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    The application uses `subprocess` to launch the main app. Run the login script first:
    ```bash
    streamlit run app.py
    ```
    [cite_start]*(Note: Based on the source code, the login script executes `streamlit run app.py` upon successful authentication [cite: 336])*

## 📊 Screenshots
* [cite_start]**Home/Login Screen:** Simple user interface for entry[cite: 347].
* [cite_start]**Prediction Interface:** Input fields for max/min temperature, precipitation, and wind speed[cite: 347].
* [cite_start]**Data Visualization:** Various charts analyzing historical trends[cite: 347].

## 🔮 Future Scope
* [cite_start]**Hybrid Algorithms:** Combining deep learning and machine learning algorithms for better accuracy[cite: 356].
* [cite_start]**Real-time Integration:** Incorporating real-time weather data feeds and advanced sensor technologies[cite: 347].
* [cite_start]**Satellite Imagery:** Exploring the use of geographical data and satellite imagery for local weather phenomena[cite: 347].

