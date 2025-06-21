from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Paths
DATASET_PATH = "balanced_ml_food_dataset.csv"
MODEL_PATH = "food_model_true_ml.pkl"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Encode categorical labels
goal_encoder = LabelEncoder()
diet_encoder = LabelEncoder()
df['Goal_Code'] = goal_encoder.fit_transform(df['Goal'])
df['Diet_Code'] = diet_encoder.fit_transform(df['Diet Type'])

# Train model
def train_model():
    print("🚧 Model is training, don't touch it! 😤")

    augmented_data = []
    for _, row in df.iterrows():
        for goal in goal_encoder.classes_:
            for diet in diet_encoder.classes_:
                label = int(row['Goal'] == goal and row['Diet Type'] == diet)
                augmented_data.append({
                    'unit_serving_energy_kcal': row['unit_serving_energy_kcal'],
                    'unit_serving_protein_g': row['unit_serving_protein_g'],
                    'unit_serving_fat_g': row['unit_serving_fat_g'],
                    'unit_serving_carb_g': row['unit_serving_carb_g'],
                    'unit_serving_fibre_g': row['unit_serving_fibre_g'],
                    'Goal': goal,
                    'Diet Type': diet,
                    'label': label
                })

    ml_df = pd.DataFrame(augmented_data)
    ml_df['Goal_Code'] = goal_encoder.transform(ml_df['Goal'])
    ml_df['Diet_Code'] = diet_encoder.transform(ml_df['Diet Type'])

    X = ml_df[['unit_serving_energy_kcal', 'unit_serving_protein_g', 'unit_serving_fat_g',
               'unit_serving_carb_g', 'unit_serving_fibre_g', 'Goal_Code', 'Diet_Code']]
    y = ml_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=48)
    model.fit(X_train, y_train)
    model_accuracy = accuracy_score(y_test, model.predict(X_test))

    print(f"✅ Model trained with accuracy: {model_accuracy:.2%}")

    if model_accuracy >= 0.94:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump((model, goal_encoder, diet_encoder, model_accuracy), f)
        return model, goal_encoder, diet_encoder, model_accuracy
    else:
        raise ValueError(f"Model accuracy only {model_accuracy:.2%}, below 94%")

# Load or train model
if not os.path.exists(MODEL_PATH):
    model, goal_encoder, diet_encoder, model_accuracy = train_model()
else:
    model, goal_encoder, diet_encoder, model_accuracy = pickle.load(open(MODEL_PATH, "rb"))
    if model_accuracy < 0.94:
        model, goal_encoder, diet_encoder, model_accuracy = train_model()

# BMI Calculation
def calculate_bmi(weight, height):
    return round(weight / ((height / 100) ** 2), 2)

# Daily Calorie Limit Calculation
def calculate_calorie_limit(weight, height, age, gender, goal):
    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == 'male' else -161)
    multiplier = {'weight loss': 0.9, 'maintenance': 1.0, 'muscle gain': 1.2}
    return round(bmr * multiplier[goal.lower()])

# ✅ Final Fixed Recommendation Function (Strict Diet Match)
def recommend_foods(goal, diet, calorie_limit):
    goal_code = goal_encoder.transform([goal])[0]
    diet_code = diet_encoder.transform([diet])[0]

    valid_foods = []
    used_food_names = set()

    for _, row in df.iterrows():
        food_name = row['food_name']
        food_diet_type = str(row['Diet Type']).strip().lower()
        user_diet = diet.strip().lower()

        if food_name in used_food_names:
            continue

        # ✅ Strictly ensure matching diet type
        if food_diet_type != user_diet:
            continue

        features = [[
            row['unit_serving_energy_kcal'],
            row['unit_serving_protein_g'],
            row['unit_serving_fat_g'],
            row['unit_serving_carb_g'],
            row['unit_serving_fibre_g'],
            goal_code,
            diet_code
        ]]
        prediction = model.predict(features)[0]

        if prediction == 1:
            valid_foods.append(row)
            used_food_names.add(food_name)

    if not valid_foods:
        return [], 0

    food_df = pd.DataFrame(valid_foods)

    # Sort foods based on goal
    if goal.lower() == "muscle gain":
        food_df["score"] = food_df["unit_serving_protein_g"] / (food_df["unit_serving_energy_kcal"] + 1e-6)
    elif goal.lower() == "weight loss":
        food_df["score"] = 1 / (food_df["unit_serving_energy_kcal"] + 1e-6)
    else:
        food_df["score"] = food_df["unit_serving_protein_g"]

    food_df = food_df.sort_values(by="score", ascending=False)

    selected = []
    total_kcal = 0

    for _, row in food_df.iterrows():
        kcal = row['unit_serving_energy_kcal']
        if total_kcal + kcal <= calorie_limit:
            food = row.to_dict()
            food['unit_serving_energy_kcal'] = round(food['unit_serving_energy_kcal'], 1)
            food['unit_serving_protein_g'] = round(food['unit_serving_protein_g'], 1)
            food['unit_serving_fat_g'] = round(food['unit_serving_fat_g'], 1)
            food['unit_serving_carb_g'] = round(food['unit_serving_carb_g'], 1)
            food['unit_serving_fibre_g'] = round(food['unit_serving_fibre_g'], 1)
            selected.append(food)
            total_kcal += kcal

        if total_kcal >= calorie_limit * 0.95:
            break

    return selected, round(total_kcal, 1)

# Flask routes
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        age = int(request.form["age"])
        gender = request.form["gender"].lower()
        goal = request.form["goal"]
        diet = request.form["diet"]

        bmi = calculate_bmi(weight, height)
        calorie_limit = calculate_calorie_limit(weight, height, age, gender, goal)
        recommendations, total_kcal = recommend_foods(goal, diet, calorie_limit)

        return render_template("index.html", bmi=bmi, calorie_limit=calorie_limit,
                               recommendations=recommendations, total_kcal=total_kcal)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)






















# UPDATED CODE 



