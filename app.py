import pickle
import gradio as gr
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# prediction logic
def prediction(age, bmi, children, sex, smoker, region):
    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region
    }])
    
    predict = model.predict(input_df)[0]
    return f"Predicted Medical Insurance Charge: {predict:.2f}"

# Gradio interface
interface = gr.Interface(
    fn=prediction,
    inputs=[
        gr.Number(label="Age", value=30, precision=0),
        gr.Number(label="BMI", value=25.0),
        gr.Number(label="Number of Children", value=0, precision=0),
        gr.Dropdown(choices=["male", "female"], label="Sex", value="male"),
        gr.Dropdown(choices=["yes", "no"], label="Smoker", value="no"),
        gr.Dropdown(choices=["northeast", "northwest", "southeast", "southwest"], label="Region", value="southeast")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Medical Insurance Cost Prediction"
)

# Launch the interface
interface.launch()
