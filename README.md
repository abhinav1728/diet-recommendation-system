# 🥗 AI-Based Diet Recommendation App

This is a Flask-based web application that provides personalized **dietary recommendations** based on a user’s **BMI, dietary preference**, and **health goal** (weight loss, muscle gain, or maintenance). It uses a **machine learning model (Random Forest Classifier)** trained on a balanced food dataset to recommend meals that match the user's nutritional needs and calorie limit.

---

## 🚀 Features

- Calculates **BMI** and **daily calorie requirements**
- Supports 3 health goals: `Weight Loss`, `Muscle Gain`, and `Maintenance`
- Supports multiple dietary preferences: `Veg`, `Non-Veg`, and `Veg + Egg`
- Intelligent filtering to strictly match user’s diet and goal
- ML-powered food prediction based on:
  - Calories (kcal)
  - Protein
  - Fat
  - Carbohydrates
  - Fiber

---

## 🧠 Machine Learning

- **Model**: RandomForestClassifier
- **Accuracy**: Trained to maintain **≥ 94% accuracy**
- **Augmented training**: Includes synthetic label generation for better prediction
- **Feature set**:
  - Energy (kcal)
  - Protein, Fat, Carbohydrates, Fiber
  - Encoded Goal & Diet Type

---

## 📁 Dataset

- Dataset: `balanced_ml_food_dataset.csv`
- Contains nutrition facts for a wide range of Indian foods
- Categorized by diet type and optimized for goal-specific filtering

---

## 💻 How It Works

1. User submits:
   - Weight, Height, Age, Gender
   - Health Goal
   - Diet Type
2. BMI and Calorie Limit are calculated
3. Model filters food items matching:
   - Nutritional need
   - Diet Type
   - Calorie limit
4. Displays a list of top foods (sorted by goal-oriented score)

---

## 🧪 Tech Stack

- **Frontend**: HTML (Jinja2 templates via Flask)
- **Backend**: Python, Flask
- **ML**: scikit-learn (Random Forest)
- **Data Handling**: pandas, NumPy

---

## 🛠 How to Run Locally

```bash
git clone https://github.com/abhinav1728/diet-recommendation-system.git
cd diet-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
