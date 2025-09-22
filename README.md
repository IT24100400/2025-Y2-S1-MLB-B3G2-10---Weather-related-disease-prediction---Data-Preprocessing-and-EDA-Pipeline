# 2025-Y2-S1-MLB-B3G2-10---Weather-related-disease-prediction---Data-Preprocessing-and-EDA-Pipeline
IT2011 - Artificial Intelligence and Machine Learning - Group Assignment

Overview of the Project

This project focuses on Weather-Related Disease Prediction, a machine learning-based initiative to predict diseases (e.g., Heart Attack, Migraine, or other conditions) based on environmental factors (like temperature, humidity, and wind speed) combined with patient symptoms (e.g., nausea, joint pain, fever) and demographics (e.g., age, gender). The goal is to build a predictive model that can assist in early diagnosis by analyzing how weather influences health outcomes.
Key stages of the project include:
  
  •	Data Cleaning and Preprocessing: Handling missing values, outliers, normalization of labels, and ensuring data quality.
  
  •	Feature Engineering: Encoding categorical variables, scaling numerical features, and selecting relevant features using techniques like variance thresholding and chi-square tests.
  
  •	Exploratory Data Analysis (EDA): Visualizing distributions, class balances, and correlations.
  
  •	Model Preparation: Preparing the dataset for training machine learning models (though the provided notebooks focus primarily on preprocessing; modeling might be in subsequent steps).
  
  •	Pipeline Integration: Combining all steps into a unified workflow for reproducibility.

The project uses Python libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn. It's designed to handle imbalanced classes (e.g., "Other" diseases dominate) and ensure the dataset is clean for downstream tasks like classification.
The final output is a processed CSV file (e.g., weather_disease_final.csv) ready for model training.

Dataset Details

The dataset is named Weather-related disease prediction (originally from a CSV file like Weather-related disease prediction (1).csv or weather_disease_cleaned-1.csv). It appears to be a synthetic or augmented dataset for educational purposes, focusing on how weather conditions correlate with disease symptoms.

Key Statistics:

  •	Shape: Approximately 4,981 rows and 52-55 columns (after preprocessing; original might have ~5,200 rows before cleaning).
  
  •	Columns:
    o	Demographics: Age (numerical, range: 1-100), Gender (binary: 0 for female, 1 for male).
    o	Weather Features (numerical/continuous): Temperature (C) (range: -15.125 to 40.996), Humidity (range: 0.370833 to 1.0), Wind Speed (km/h) (range: 0.008 to 31.303).
    o	Symptoms (binary: 0/1 indicators): ~45 columns like nausea, joint_pain, abdominal_pain, high_fever, chills, shortness_of_breath, reduced_smell_and_taste, skin_irritation, itchiness, throbbing_headache, confusion, back_pain, knee_ache, etc.
    
    o	Target Variables: 
      	prognosis: Original multi-class label (e.g., "Heart Attack", "Influenza", "Dengue", "Sinusitis", "Migraine", etc.).
      	prognosis_encoded: Label-encoded version of prognosis.
      	prognosis_3class: Simplified 3-class label (normalized and mapped): 
      	"Heart Attack" (~1,013 instances).
      	"Migraine" (~941 instances).
      	"Other" (~3,246 instances, including Influenza, Dengue, etc.).
      	One-hot encoded disease columns (e.g., disease_Heart Attack, disease_Migraine, etc.) for multi-class handling.
  •	Data Types: Mostly numerical (floats for weather, integers for binaries and age), with some strings for targets before encoding.
  •	Challenges: 
      o	Imbalanced classes: "Other" dominates (~65%), while "Heart Attack" and "Migraine" are minorities.
      o	Missing Values: Handled by imputation (e.g., mean for continuous, mode for binary).
      o	Outliers: Removed using IQR method (e.g., reduced rows from 5,200 to 4,981).
      o	Low Variance Features: Removed if variance < 0.0.
      o	Scaling: Applied MinMaxScaler to normalize continuous features to [0,1].
  •	Sources/Notes: The dataset is loaded from CSV files in the notebooks. It's cleaned progressively, with final versions like weather_disease_onehot_scaled.csv, weather_disease_lowvar.csv, weather_disease_chi_Square.csv, and weather_disease_final.csv.
  
Class Distribution (from EDA):

  •	Other: 3,246
  •	Heart Attack: 1,013
  •	Migraine: 941

Group Member Roles

Based on the notebook filenames (which include student IDs like "IT24100XXX"), this appears to be a group project where each member handled a specific preprocessing or feature engineering task. Here's an inferred breakdown:
•	IT24100477: Handled Encoding (LabelEncoder for targets, one-hot encoding for multi-class prognosis). Created visualizations like bar plots for class distribution.
•	IT24100400: Responsible for MinMax Scaling and Box Plots. Applied scaling to normalize continuous features (e.g., Age, Temperature) and visualized distributions with box plots before/after scaling.
•	IT24100492: Focused on Variance Threshold Feature Selection. Removed low-variance features (threshold=0.0) to eliminate redundant columns.
•	IT24100502: Managed Chi-Square Feature Selection. Used SelectKBest with chi2 to select top features based on statistical dependence with the target.
•	IT24100423: Dealt with Outlier Removal. Used IQR method to detect and remove outliers in continuous columns (e.g., Age, Temperature), reducing dataset size and providing before/after summaries.
•	IT24100562: Handled Missing Data. Performed EDA on missing values (heatmaps, histograms), imputed using mean/mode strategies, and visualized class balances.

•	Group Pipeline (No Specific ID): Integrated all steps into a single pipeline notebook. This combines loading, cleaning, encoding, scaling, feature selection, and saving the final dataset. It uses tools like SimpleImputer for missing values and joblib for model persistence.
The group likely collaborated on the overall pipeline, with each member contributing their specialized notebook.



How to Run the Code

These are Google Colab environment. They require Python 3+ and libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn (install via pip install pandas numpy matplotlib seaborn scikit-learn).

Steps to Run:
 1.	Setup Environment: 
    o	Google Colab (Recommended, as notebooks use from google.colab import files for uploads): 
 2.	Dependencies: 
    o	Ensure libraries are installed (most notebooks import them at the top).
    o	No external APIs or advanced setups needed. 
 3.	Running Individual Notebooks: 
    o	Open a notebook (e.g., IT24100477Encoding.ipynb).
    o	Run cells sequentially (Shift+Enter).
    o	When prompted, upload the input CSV (e.g., weather_disease_cleaned-1.csv) using the file uploader.
    o	Outputs include prints (e.g., data shapes, missing values), plots (e.g., heatmaps, bar plots), and saved CSVs (e.g., via df.to_csv()).
    o	Note: Some notebooks depend on outputs from others (e.g., scaling uses encoded data), so run in logical order: Missing Data → Outlier Removal → Encoding → Scaling → Variance Threshold → Chi-Square.
 4.	Running the Group Pipeline (Group_Pipeline_.ipynb): 
    o	This is the main entry point.
    o	Upload the initial dataset CSV when prompted.
    o	Run all cells: It loads data, handles missing values (SimpleImputer), removes outliers (IQR), encodes labels (LabelEncoder), scales features (MinMaxScaler), selects features (VarianceThreshold + SelectKBest/chi2), and saves the final CSV.
    o	Outputs: Processed DataFrame previews, visualizations, and a downloadable weather_disease_final.csv.
    o	To download results: The notebook uses files.download() for CSVs.
 5.	Common Tips/Troubleshooting: 
    o	File Paths: Notebooks assume Google Drive/Colab uploads; adjust paths if running locally (e.g., change to pd.read_csv('local/path.csv')).
    o	Errors: If a library is missing, add !pip install <library> in a cell.
    o	Runtime: Quick (seconds per notebook) since dataset is small (~5K rows).
    o	Reproducibility: Set random seeds if needed (not explicitly in notebooks).
    o	Outputs: Visuals (plots) appear inline; CSVs are saved and downloadable

