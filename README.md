# IPL Score Prediction

## 🏏 Project Overview

IPL Score Prediction is a machine learning-based project that predicts the **final score of an IPL innings** based on the current match situation.

The system takes important match details such as the batting team, bowling team, venue, current runs, wickets, overs, and recent performance, and uses a trained machine learning model to estimate the expected final score.

The project demonstrates how machine learning can be applied to sports analytics and real-world cricket data to make data-driven predictions.

---

## 🎯 Objective

* Predict the probable final score of an IPL innings
* Use historical IPL match data for machine learning
* Analyze the current match situation and recent performance
* Build a simple web-based interface for score prediction
* Demonstrate the application of machine learning in sports analytics

---

## 🧠 Algorithm Used

## Machine Learning – XGBoost

The project uses **XGBoost**, a powerful gradient boosting machine learning algorithm, to predict the final score of an IPL innings.

The model is trained using historical IPL data and learns the relationship between match conditions and the final innings score.

During prediction, the model considers factors such as the current score, wickets lost, overs completed, teams, venue, and performance during the previous five overs to estimate the final score.

---

## ⚙️ Features

* 🏏 IPL innings score prediction
* 👥 Batting and bowling team selection
* 🏟️ Venue-based prediction
* ⏱️ Current over and match situation input
* 📊 Current runs and wickets tracking
* 📈 Last 5 overs performance analysis
* 🤖 Machine learning-based prediction
* 🌐 Flask-based web application
* ⚡ Quick prediction through an interactive interface

---

## 🛠️ Technologies Used

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* HTML
* CSS
* JavaScript

---

## 📂 Project Structure

```text
IPL_Prediction/
│── app.py
│── model/
│   ├── ipl_score_model.pkl
│   └── encoders.pkl
│── static/
│── templates/
│   └── index.html
│── requirements.txt
│── README.md
```
