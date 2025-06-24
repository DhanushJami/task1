# task1
# ETL Pipeline: Balaji Fast Food Sales Data

This project implements an automated **ETL (Extract, Transform, Load) pipeline** for processing and transforming sales data using **pandas** and **scikit-learn**. The pipeline cleans, preprocesses, and transforms raw data, then splits it into training and testing sets and saves the processed outputs as CSV files.

---

## 🚀 Features

- Loads CSV data
- Drops irrelevant or high-cardinality columns
- Handles missing values (imputation)
- Scales numeric features
- Encodes categorical variables using one-hot encoding
- Splits data into train and test sets (80/20)
- Saves preprocessed data to CSV files
- Prints sample data and status in the terminal

---

## 🧠 Technologies Used

- Python
- pandas
- scikit-learn

---

## 📁 Input File

Ensure that the file named **`Balaji Fast Food Sales.csv`** is placed in the same directory as `app.py`.

---

## 📜 How to Run

### 1. Install Required Libraries

Make sure you have the required Python packages:

```bash
pip install pandas scikit-learn
