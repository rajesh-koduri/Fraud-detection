# Financial Transactions Fraud Detection

## Project Overview

This project aims to build and evaluate a machine learning model to detect fraudulent financial transactions. It utilizes a dataset containing various transaction details and employs a RandomForestClassifier. Special attention is given to addressing class imbalance, a common issue in fraud detection datasets, through techniques like SMOTE (Synthetic Minority Over-sampling Technique).

## Dataset

The dataset `financial_transactions.csv` contains simulated financial transaction data with features such as `transaction_amount`, `transaction_time`, `transaction_type`, `account_age_days`, and a `fraud` label indicating whether a transaction is fraudulent (1) or not (0).

## Key Features

-   **Data Loading and Exploration**: Loading transaction data into a Pandas DataFrame and performing initial exploration.
-   **Data Preprocessing**: Encoding categorical features (`transaction_type`, `day_of_week`) using `LabelEncoder`.
-   **Data Splitting**: Dividing the dataset into training and testing sets.
-   **Model Training**: Training a RandomForestClassifier for fraud detection.
-   **Model Evaluation**: Assessing model performance using metrics like Confusion Matrix, Classification Report (Precision, Recall, F1-score), and ROC AUC Score.
-   **Class Imbalance Handling**: Implementing SMOTE to address the imbalance between fraudulent and non-fraudulent transactions in the training data.
-   **Prediction**: Demonstrating how to use the trained model to predict fraud for new, unseen transactions.

## Technologies Used

-   Python
-   Pandas (for data manipulation)
-   NumPy (for numerical operations)
-   Matplotlib & Seaborn (for data visualization)
-   Scikit-learn (for machine learning models and utilities)
-   Imblearn (for handling imbalanced datasets, specifically SMOTE)

## Setup and Usage (Google Colab)

This notebook is designed to run in Google Colab.

1.  **Upload Data**: Ensure `financial_transactions.csv` is uploaded to your Colab environment (e.g., to the `/content/` directory or mounted Google Drive).
2.  **Run Cells**: Execute the notebook cells sequentially.
    -   The initial cells load data, perform basic exploration, and visualize transaction types against fraud rates.
    -   Subsequent cells handle data preprocessing, model training (both without and with SMOTE), and evaluation.
    -   The final cells demonstrate how to make predictions on new data.

## Model Performance

Initial model evaluation revealed a significant challenge: the model struggled to identify fraudulent transactions, achieving very low recall and F1-scores for the fraud class. This is characteristic of highly imbalanced datasets where the minority class (fraud) is rare.

To mitigate this, SMOTE was applied to oversample the minority class in the training data. The re-trained model's performance on the original (unbalanced) test set is then re-evaluated, showing improved ability to detect fraudulent transactions (increased recall) while managing precision.

### Evaluation Metrics Focused On:

-   **Recall**: Crucial for fraud detection, as missing fraudulent transactions can be costly. We aim to maximize the detection of actual fraud cases.
-   **F1-score**: A harmonic mean of precision and recall, providing a balanced measure.
-   **ROC AUC Score**: Measures the model's ability to distinguish between classes.

## Future Improvements

-   **Hyperparameter Tuning**: Optimize the `RandomForestClassifier` or other models using techniques like GridSearchCV or RandomizedSearchCV.
-   **Feature Engineering**: Create more informative features from existing data.
-   **Anomaly Detection Algorithms**: Explore algorithms specifically designed for anomaly detection, such as Isolation Forest or One-Class SVM.
-   **Advanced Resampling**: Investigate other imblearn techniques or ensemble methods tailored for imbalance (e.g., EasyEnsembleClassifier, BalancedBaggingClassifier).
-   **Cost-Sensitive Learning**: Incorporate the differing costs of false positives and false negatives into the model training process.

## License

This project is open-sourced under the MIT License.
