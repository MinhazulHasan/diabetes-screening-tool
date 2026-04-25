import gradio as gr
import joblib
import pandas as pd
import numpy as np

model = joblib.load("diabetes_model.pkl")
meta = joblib.load("metadata.pkl")

ALL_FEATURES = meta["all_features"]
MEDIANS = meta["medians"]
METRICS = meta.get("final_metrics", {})


def bmi_category(bmi):
    if pd.isna(bmi) or bmi == 0: return 'Unknown'
    if bmi < 18.5: return 'Underweight'
    if bmi < 25: return 'Normal'
    if bmi < 30: return 'Overweight'
    return 'Obese'


def age_group(age):
    if age < 30: return 'Young'
    if age < 45: return 'Middle'
    if age < 60: return 'Senior'
    return 'Elderly'


def glucose_range(g):
    if pd.isna(g) or g == 0: return 'Unknown'
    if g < 100: return 'Normal'
    if g < 126: return 'Prediabetic'
    return 'Diabetic_range'


WAITING_HTML = """
<div style='padding: 40px 24px; border-radius: 14px; background: #f1f5f9;
            border: 2px dashed #cbd5e1; text-align: center; min-height: 320px;
            display: flex; flex-direction: column; justify-content: center;'>
    <div style='font-size: 56px; color: #94a3b8; margin-bottom: 12px;'>—</div>
    <p style='color: #64748b; font-size: 16px; margin: 0;'>
        Fill in the patient details and click <b>Predict</b><br>to see the result here.
    </p>
</div>
"""


def predict(Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age):
    row = {
        'Pregnancies': float(Pregnancies),
        'Glucose': float(Glucose) if Glucose > 0 else np.nan,
        'BloodPressure': float(BloodPressure) if BloodPressure > 0 else np.nan,
        'SkinThickness': float(SkinThickness) if SkinThickness > 0 else np.nan,
        'Insulin': float(Insulin) if Insulin > 0 else np.nan,
        'BMI': float(BMI) if BMI > 0 else np.nan,
        'DiabetesPedigreeFunction': float(DiabetesPedigreeFunction),
        'Age': float(Age),
    }

    row['BMI_Category'] = bmi_category(row['BMI'])
    row['Age_Group'] = age_group(row['Age'])
    row['Glucose_Range'] = glucose_range(row['Glucose'])
    bmi_for_score = row['BMI'] if not pd.isna(row['BMI']) else MEDIANS.get('BMI', 32)
    row['HighRisk_Score'] = (row['Age'] / 100) * (bmi_for_score / 40) * (row['DiabetesPedigreeFunction'] * 2)
    row['Insulin_BMI_Ratio'] = row['Insulin'] / row['BMI'] if not pd.isna(row['Insulin']) and not pd.isna(row['BMI']) else np.nan

    df_input = pd.DataFrame([row])[ALL_FEATURES]
    proba = float(model.predict_proba(df_input)[0][1])
    pred = int(model.predict(df_input)[0])
    pct = proba * 100

    if pred == 1:
        label = "Diabetic"
        color = "#dc2626"
        bg = "#fef2f2"
        accent = "Likely diabetic. A confirmatory clinical test is recommended."
    else:
        label = "Non-Diabetic"
        color = "#16a34a"
        bg = "#f0fdf4"
        accent = "Low risk based on the entered measurements."

    if pct >= 60:
        risk_label, risk_color = "HIGH", "#dc2626"
    elif pct >= 30:
        risk_label, risk_color = "MODERATE", "#d97706"
    else:
        risk_label, risk_color = "LOW", "#16a34a"

    html = f"""
    <div style='padding: 28px 24px; border-radius: 14px; background: {bg};
                border-left: 6px solid {color}; min-height: 320px;'>
        <div style='display: flex; justify-content: space-between; align-items: center;
                    margin-bottom: 18px;'>
            <h2 style='color: {color}; margin: 0; font-size: 26px;'>{label}</h2>
            <span style='background: {risk_color}; color: white; padding: 6px 14px;
                         border-radius: 999px; font-weight: 600; font-size: 13px;
                         letter-spacing: 0.5px;'>{risk_label} RISK</span>
        </div>

        <div style='text-align: center; padding: 18px 0; border-top: 1px solid #e2e8f0;
                    border-bottom: 1px solid #e2e8f0; margin-bottom: 18px;'>
            <p style='color: #64748b; font-size: 13px; margin: 0 0 6px 0;
                      text-transform: uppercase; letter-spacing: 0.8px;'>
                Probability of diabetes
            </p>
            <div style='font-size: 56px; font-weight: 700; color: {color}; line-height: 1;'>
                {pct:.1f}%
            </div>
        </div>

        <div style='background: white; padding: 14px 16px; border-radius: 8px;
                    border: 1px solid #e2e8f0;'>
            <p style='color: #334155; margin: 0; font-size: 14px; line-height: 1.55;'>
                {accent}
            </p>
        </div>
    </div>
    """
    return html


theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    button_primary_background_fill="#1e3a8a",
    button_primary_background_fill_hover="#1e40af",
    button_primary_text_color="white",
    slider_color="#1e3a8a",
    body_background_fill="#f8fafc",
)


with gr.Blocks(theme=theme, title="Diabetes Prediction") as demo:
    gr.Markdown("# Diabetes Prediction")
    gr.Markdown(
        "Enter the patient measurements on the left, then click **Predict** "
        "to see the result on the right. Set a value to 0 if it is unknown."
    )

    with gr.Row():
        # LEFT COLUMN — Inputs
        with gr.Column(scale=3):
            Pregnancies = gr.Slider(0, 17, value=3, step=1,
                                     label="Number of Pregnancies (count)")
            Glucose = gr.Slider(0, 199, value=120, step=1, label="Glucose (mg/dL)")
            BloodPressure = gr.Slider(0, 122, value=70, step=1, label="Blood Pressure (mm Hg)")
            SkinThickness = gr.Slider(0, 99, value=25, step=1, label="Skin Thickness (mm)")
            Insulin = gr.Slider(0, 846, value=80, step=5, label="Insulin (µU/mL)")
            BMI = gr.Slider(15, 67.1, value=30, step=0.1, label="BMI (kg/m²)")
            DiabetesPedigreeFunction = gr.Slider(0.078, 2.42, value=0.5, step=0.01,
                                                  label="Diabetes Pedigree Function (family history score, 0.08 to 2.42)")
            Age = gr.Slider(21, 81, value=33, step=1, label="Age (years)")

            submit = gr.Button("Predict", variant="primary", size="lg")

        # RIGHT COLUMN — Result
        with gr.Column(scale=2):
            output = gr.HTML(value=WAITING_HTML, label="Result")

    submit.click(
        fn=predict,
        inputs=[Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age],
        outputs=output,
    )

    gr.Markdown(
        f"""
        ---
        **Model performance on test set:**
        Accuracy {METRICS.get('Accuracy', 0.7662)*100:.2f}% &nbsp;|&nbsp;
        Precision {METRICS.get('Precision', 0.6324)*100:.2f}% &nbsp;|&nbsp;
        Recall {METRICS.get('Recall', 0.7963)*100:.2f}% &nbsp;|&nbsp;
        F1 {METRICS.get('F1-score', 0.7049)*100:.2f}% &nbsp;|&nbsp;
        ROC-AUC {METRICS.get('ROC-AUC', 0.8348)*100:.2f}%
        """
    )


if __name__ == "__main__":
    demo.launch()
