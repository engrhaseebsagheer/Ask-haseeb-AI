# Earlier work: Haseeb Sagheer's 2025 machine learning projects

These five projects are from 2025, when Haseeb was studying data science and machine learning. They are earlier work, not his current focus. His current work is AI automation and his own products. The live demos of these projects are now offline; the code for each is on GitHub.

## Fake News Detector (2025)
A web application that classifies a news article as real or fake.
- Data: the Kaggle Fake News Detection dataset (fake.csv and true.csv).
- Method: text cleaning and lemmatisation, TF-IDF features, and a comparison of Logistic Regression, Random Forest, Naive Bayes and LinearSVC. LinearSVC was chosen for speed and accuracy.
- Delivery: a FastAPI app with a web page and a /predict API, deployed on an Ubuntu VPS with Gunicorn, Apache and HTTPS.
- Built with: Python, scikit-learn, pandas, FastAPI.
- Code: https://github.com/engrhaseebsagheer/Fake-News-Detector
- Status: demo offline.

## Telco Customer Churn Prediction (August 2025)
A machine learning project that predicts which telecom customers are likely to leave.
- Data: customer demographics, contract type, services and billing.
- Method: data cleaning, encoding, a stratified train and test split, then Logistic Regression and Random Forest.
- Result: 80% accuracy and a ROC-AUC of 0.84 for both models.
- Findings: short tenure, month-to-month contracts and fibre optic internet were linked to higher churn.
- Built with: Python, pandas, scikit-learn, seaborn, matplotlib.
- Code: https://github.com/engrhaseebsagheer/customer-churn
- Status: demo offline.

## Titanic Survival Predictor (2025)
A baseline model for the Kaggle Titanic competition that predicts passenger survival.
- Method: data exploration, imputing missing ages, encoding categories, Logistic Regression with a train and validation split, and a Kaggle submission.
- Built with: Python, pandas, scikit-learn, seaborn.
- Code: https://github.com/engrhaseebsagheer/titanic-survival-model
- Status: demo offline.

## Professional CV Generator (2025)
A Flask web app where a user fills in a form and downloads a formatted CV as a PDF.
- Sections: personal details, work experience, education, skills, projects and links.
- Built with: Python, Flask, FPDF, HTML and CSS.
- Code: https://github.com/engrhaseebsagheer/Professional-CV-Generator
- Status: demo offline.

## Simple Linear Regression from Scratch (2025)
Linear regression written in pure Python with no machine learning libraries, to understand the mathematics.
- A SimpleLinearRegression class with fit, predict, R squared, MSE and RMSE, using the closed-form equation.
- Result on the sample dataset: R squared of 0.99 on training data and 0.70 on validation data.
- Code: https://github.com/engrhaseebsagheer/Linear-Regression-From-Scratch
- Status: demo offline.

## Does Haseeb still do machine learning and data science work?
He has the background: the IBM Data Science Professional Certificate and the projects above. His current work is AI automation, RAG assistants, LLM features and building products. See his current projects for what he builds now.

## Are the old demos on haseebsagheer.com still online?
No. The fake news detector, churn predictor, Titanic model, CV generator and linear regression demos were taken offline when the site was rebuilt in 2026. The code remains on GitHub.
