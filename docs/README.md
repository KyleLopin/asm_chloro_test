# asm_chloro_test

This project predicts chlorophyll concentration in leaves from spectral sensor data using machine learning models.  

This project includes:
- Arduino code to control AMS sensors (AS7262, AS7263, AS7265x) and collect spectral data. 
- A Python-based GUI application to interface with the sensors.
- STL files for 3D printing shrouds and holders for the sensor modules.
- Experimental datasets measuring chlorophyll levels across five types of leaves: banana, jasmine, mango, sugarcane, and rice.
- Analysis files.

---

## Project Structure

- **source/**
  - Arduino firmware to control AMS sensors.
  - Python GUI application for sensor control and data saving.

- **3d_files/**
  - STL files for 3D printing button and sensor shrouds and holders.

- **data/**
  - Contains input sensor data files and processed datasets.
  - Includes chlorophyll concentration measurements and corresponding spectroscopic readings.

- **analysis/**
  - Contains main analysis and testing scripts.
  - Includes functions for:
    - Data preprocessing (removing chlorophyll and spectral outliers)
    - Model training (e.g., Linear Regression, PLS)
    - Hyperparameter fitting
    - Cross-validation and error analysis

---

## Key Features

- Choose best sensor and model settings (e.g., integration time, led current) using N-Way ANOVA.
- Remove outliers using Mahalanobis distance of spectra
- Implement regression models for chlorophyll prediction.
- Tune Partial Least Squared number of components by Akaike Information Criterion
- Perform cross-validation and calculate error metrics.
- Analyze how different preprocessing impacts model performance.

---

## Libraries Used

- Pandas
- Scikit-Learn
- Matplotlib

---

## Goals

- Optimize preprocessing and modeling pipeline for accurate chlorophyll prediction.
- Develop methods that could be generalized for other types of spectral analysis.

---

## Author

Kyle Lopin
