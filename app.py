import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Generate synthetic data (simulate Indian house prices)
np.random.seed(42)
n = 1000
income = np.random.uniform(2, 20, n)  # income in lakhs per year
rooms = np.random.randint(1, 6, n)
price = (income * 5 + rooms * 10 + np.random.normal(0, 10, n)) * 1e5  # price in INR

# Create DataFrame
df = pd.DataFrame({
    "income_lakhs": income,
    "rooms": rooms,
    "price_inr": price
})

# Train model
X = df[["income_lakhs", "rooms"]]
y = df["price_inr"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediction function
def predict_house_price(income_lakhs, rooms):
    input_df = pd.DataFrame([[income_lakhs, rooms]], columns=["income_lakhs", "rooms"])
    price = model.predict(input_df)[0]
    return f"🏡 Estimated Price: ₹{int(price):,}"

# Gradio UI
interface = gr.Interface(
    fn=predict_house_price,
    inputs=[
        gr.Slider(2, 20, step=0.5, label="Household Income (in ₹ Lakhs/year)"),
        gr.Slider(1, 5, step=1, label="Number of Rooms")
    ],
    outputs=gr.Text(label="Predicted Price"),
    title="🇮🇳 Indian House Price Predictor",
    description="Predicts house price in INR based on income and number of rooms."
)

interface.launch()
