
import gradio as gr
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Load model, scaler and saved files
model = joblib.load("../models/logistic_regression_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

df = pd.read_csv("../data/heart_disease_prepared.csv")
model_results = pd.read_csv("../models/model_comparison_results.csv")
shap_importance = pd.read_csv("../models/shap_feature_importance.csv")

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def create_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        number={"suffix": "%"},
        title={"text": "Predicted Heart Disease / Attack Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#E94F37"},
            "steps": [
                {"range": [0, 40], "color": "#D5F5E3"},
                {"range": [40, 70], "color": "#FCF3CF"},
                {"range": [70, 100], "color": "#FADBD8"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": probability
            }
        }
    ))

    fig.update_layout(
        template="plotly_white",
        height=300,
        margin=dict(l=30, r=30, t=50, b=20)
    )

    return fig

def create_risk_flags_chart(risk_flags):
    if len(risk_flags) == 0:
        chart_df = pd.DataFrame({
            "Risk Factor": ["No major selected risk flags"],
            "Present": [0]
        })
    else:
        chart_df = pd.DataFrame({
            "Risk Factor": risk_flags,
            "Present": [1] * len(risk_flags)
        })

    fig = px.bar(
        chart_df,
        x="Present",
        y="Risk Factor",
        orientation="h",
        title="Selected Risk Flags",
        color="Present",
        color_continuous_scale="Reds"
    )

    fig.update_layout(
        template="plotly_white",
        height=350,
        showlegend=False,
        xaxis_title="",
        yaxis_title="",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    fig.update_xaxes(showticklabels=False)

    return fig

def predict_heart_disease(
    HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, Diabetes,
    PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare,
    NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex,
    Age, Education, Income
):
    diabetes_any = 1 if Diabetes > 0 else 0
    risk_factor_count = HighBP + HighChol + Smoker + Stroke + DiffWalk + diabetes_any
    total_poor_health_days = MentHlth + PhysHlth

    row = {
        "HighBP": HighBP, "HighChol": HighChol, "CholCheck": CholCheck, "BMI": BMI,
        "Smoker": Smoker, "Stroke": Stroke, "PhysActivity": PhysActivity, "Fruits": Fruits,
        "Veggies": Veggies, "HvyAlcoholConsump": HvyAlcoholConsump, "AnyHealthcare": AnyHealthcare,
        "NoDocbcCost": NoDocbcCost, "GenHlth": GenHlth, "MentHlth": MentHlth, "PhysHlth": PhysHlth,
        "DiffWalk": DiffWalk, "Sex": Sex, "Age": Age, "Education": Education, "Income": Income,
        "risk_factor_count": risk_factor_count,
        "total_poor_health_days": total_poor_health_days,
        "Diabetes_1.0": 1 if Diabetes == 1 else 0,
        "Diabetes_2.0": 1 if Diabetes == 2 else 0,
    }

    model_features = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
        "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex",
        "Age", "Education", "Income", "risk_factor_count", "total_poor_health_days",
        "Diabetes_1.0", "Diabetes_2.0"
    ]

    input_data = pd.DataFrame([row])[model_features]
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        risk_label = "Higher Heart Disease / Attack Risk"
    else:
        risk_label = "Lower Heart Disease / Attack Risk"

    bmi_category = get_bmi_category(BMI)

    age_map = {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39",
        5: "40-44", 6: "45-49", 7: "50-54", 8: "55-59",
        9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79",
        13: "80+"
    }

    health_map = {
        1: "Excellent",
        2: "Very Good",
        3: "Good",
        4: "Fair",
        5: "Poor"
    }

    risk_flags = []

    if HighBP == 1:
        risk_flags.append("High blood pressure")
    if HighChol == 1:
        risk_flags.append("High cholesterol")
    if Smoker == 1:
        risk_flags.append("Smoking history")
    if Stroke == 1:
        risk_flags.append("Stroke history")
    if Diabetes == 2:
        risk_flags.append("Diabetes")
    elif Diabetes == 1:
        risk_flags.append("Pre-diabetes")
    if PhysActivity == 0:
        risk_flags.append("No physical activity")
    if DiffWalk == 1:
        risk_flags.append("Difficulty walking")
    if GenHlth >= 4:
        risk_flags.append("Fair/Poor general health")
    if BMI >= 30:
        risk_flags.append("Obese BMI category")

    if probability >= 70:
        risk_badge = "🔴 Higher predicted risk"
    elif probability >= 40:
        risk_badge = "🟠 Moderate predicted risk"
    else:
        risk_badge = "🟢 Lower predicted risk"

    summary = f'''
## Prediction Result

### {risk_badge}

**Model Prediction:** {risk_label}

**Predicted Probability:** {probability:.2f}%

---

## User Health Snapshot

**BMI:** {BMI}  
**BMI Category:** {bmi_category}

**Age Group:** {age_map.get(Age, "Unknown")}

**General Health:** {health_map.get(GenHlth, "Unknown")}

**Poor Mental Health Days:** {MentHlth} / 30

**Poor Physical Health Days:** {PhysHlth} / 30

---

This result is for educational and portfolio purposes only. It is not a medical diagnosis and should not be used for clinical decision-making.
'''

    gauge = create_gauge(probability)
    risk_chart = create_risk_flags_chart(risk_flags)

    return summary, gauge, risk_chart

def create_target_distribution_graph():
    target_counts = df["HeartDiseaseorAttack"].value_counts().reset_index()
    target_counts.columns = ["HeartDiseaseorAttack", "Count"]

    target_counts["Heart Disease Status"] = target_counts["HeartDiseaseorAttack"].map({
        0.0: "No Heart Disease / Attack",
        1.0: "Heart Disease / Attack"
    })

    fig = px.bar(
        target_counts,
        x="Heart Disease Status",
        y="Count",
        text="Count",
        title="Heart Disease / Attack Distribution",
        color="Heart Disease Status",
        color_discrete_map={
            "No Heart Disease / Attack": "#2E86AB",
            "Heart Disease / Attack": "#E94F37"
        }
    )

    fig.update_traces(
        textposition="outside",
        marker_line_color="white",
        marker_line_width=1.5,
        opacity=0.9,
        hovertemplate="<b>%{x}</b><br>Number of Records: %{y}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Heart Disease Status",
        yaxis_title="Number of Records",
        showlegend=False,
        title_x=0.5,
        template="plotly_white",
        height=500
    )

    return fig

def create_age_graph():
    age_rate = df.groupby("Age")["HeartDiseaseorAttack"].mean().reset_index()
    age_rate["HeartDiseaseorAttack"] = age_rate["HeartDiseaseorAttack"] * 100

    age_rate["Age Group"] = age_rate["Age"].map({
        1.0: "18-24", 2.0: "25-29", 3.0: "30-34", 4.0: "35-39",
        5.0: "40-44", 6.0: "45-49", 7.0: "50-54", 8.0: "55-59",
        9.0: "60-64", 10.0: "65-69", 11.0: "70-74", 12.0: "75-79",
        13.0: "80+"
    })

    fig = px.line(
        age_rate,
        x="Age Group",
        y="HeartDiseaseorAttack",
        markers=True,
        title="Heart Disease / Attack Rate by Age Group",
        text=age_rate["HeartDiseaseorAttack"].round(2).astype(str) + "%"
    )

    fig.update_traces(
        line=dict(width=4, color="#8E44AD"),
        marker=dict(size=10, color="#E67E22"),
        textposition="top center",
        hovertemplate="<b>Age Group: %{x}</b><br>Rate: %{y:.2f}%<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Age Group",
        yaxis_title="Heart Disease / Attack Rate (%)",
        title_x=0.5,
        template="plotly_white",
        height=500
    )

    return fig

def create_model_comparison_graph():
    model_results_long = model_results.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        model_results_long,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text=model_results_long["Score"].round(3),
        title="Model Performance Comparison",
        color_discrete_map={
            "Logistic Regression": "#636EFA",
            "Random Forest": "#EF553B",
            "XGBoost": "#00CC96"
        }
    )

    fig.update_traces(
        textposition="outside",
        marker_line_color="white",
        marker_line_width=1.2,
        opacity=0.9,
        hovertemplate="<b>%{fullData.name}</b><br>Metric: %{x}<br>Score: %{y:.3f}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Evaluation Metric",
        yaxis_title="Score",
        title_x=0.5,
        template="plotly_white",
        height=550
    )

    return fig

def create_shap_graph():
    top_shap_features = shap_importance.head(15)

    fig = px.bar(
        top_shap_features.sort_values(by="Mean Absolute SHAP Value"),
        x="Mean Absolute SHAP Value",
        y="Feature",
        orientation="h",
        title="Top SHAP Feature Importance",
        color="Mean Absolute SHAP Value",
        color_continuous_scale="Tealgrn"
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Mean Absolute SHAP Value: %{x:.4f}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
        title_x=0.5,
        template="plotly_white",
        height=650
    )

    return fig

yes_no = [("No", 0), ("Yes", 1)]

with gr.Blocks(title="Heart Disease Risk Prediction App") as demo:

    gr.Markdown(
        '''
        # Heart Disease Risk Prediction App

        Educational machine learning app for heart disease/heart attack risk prediction using public health indicators.
        '''
    )

    with gr.Tabs():

        with gr.Tab("Risk Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Patient / User Inputs")

                    HighBP = gr.Radio(yes_no, label="High Blood Pressure", value=0)
                    HighChol = gr.Radio(yes_no, label="High Cholesterol", value=0)
                    CholCheck = gr.Radio(yes_no, label="Cholesterol Check in Last 5 Years", value=1)
                    BMI = gr.Number(label="BMI", value=25)

                    Smoker = gr.Radio(yes_no, label="Smoker", value=0)
                    Stroke = gr.Radio(yes_no, label="Stroke History", value=0)
                    Diabetes = gr.Radio(
                        [("No Diabetes", 0), ("Pre-diabetes", 1), ("Diabetes", 2)],
                        label="Diabetes Status",
                        value=0
                    )

                    PhysActivity = gr.Radio(yes_no, label="Physical Activity", value=1)
                    Fruits = gr.Radio(yes_no, label="Consumes Fruit Daily", value=1)
                    Veggies = gr.Radio(yes_no, label="Consumes Vegetables Daily", value=1)
                    HvyAlcoholConsump = gr.Radio(yes_no, label="Heavy Alcohol Consumption", value=0)

                    AnyHealthcare = gr.Radio(yes_no, label="Has Healthcare Coverage", value=1)
                    NoDocbcCost = gr.Radio(yes_no, label="Could Not See Doctor Due to Cost", value=0)

                    GenHlth = gr.Radio(
                        [("Excellent", 1), ("Very Good", 2), ("Good", 3), ("Fair", 4), ("Poor", 5)],
                        label="General Health",
                        value=2
                    )

                    MentHlth = gr.Slider(0, 30, step=1, label="Poor Mental Health Days in Last 30 Days", value=0)
                    PhysHlth = gr.Slider(0, 30, step=1, label="Poor Physical Health Days in Last 30 Days", value=0)

                    DiffWalk = gr.Radio(yes_no, label="Difficulty Walking", value=0)
                    Sex = gr.Radio([("Female", 0), ("Male", 1)], label="Sex", value=0)

                    Age = gr.Radio(
                        [
                            ("18-24", 1), ("25-29", 2), ("30-34", 3), ("35-39", 4),
                            ("40-44", 5), ("45-49", 6), ("50-54", 7), ("55-59", 8),
                            ("60-64", 9), ("65-69", 10), ("70-74", 11), ("75-79", 12),
                            ("80+", 13)
                        ],
                        label="Age Group",
                        value=8
                    )

                    Education = gr.Slider(1, 6, step=1, label="Education Level", value=4)
                    Income = gr.Slider(1, 8, step=1, label="Income Level", value=5)

                    predict_button = gr.Button("Predict Heart Disease Risk")

                with gr.Column(scale=1):
                    gr.Markdown("## Results & Mini Dashboard")

                    prediction_output = gr.Markdown()
                    gauge_output = gr.Plot()
                    risk_chart_output = gr.Plot()

            predict_button.click(
                fn=predict_heart_disease,
                inputs=[
                    HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, Diabetes,
                    PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare,
                    NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex,
                    Age, Education, Income
                ],
                outputs=[prediction_output, gauge_output, risk_chart_output]
            )

        with gr.Tab("Data Dashboard"):
            gr.Markdown("## Dataset Overview and EDA Dashboard")
            gr.Plot(value=create_target_distribution_graph())
            gr.Plot(value=create_age_graph())

        with gr.Tab("Model Performance"):
            gr.Markdown("## Model Performance Comparison")
            gr.Plot(value=create_model_comparison_graph())
            gr.Dataframe(value=model_results)

        with gr.Tab("Explainability"):
            gr.Markdown("## SHAP Explainability")
            gr.Plot(value=create_shap_graph())

        with gr.Tab("About Project"):
            gr.Markdown(
                '''
                ## About This Project

                This project predicts heart disease/heart attack risk using public health indicator data.

                The project includes:
                - Data loading and cleaning
                - Exploratory data analysis
                - Machine learning model training
                - Model comparison
                - SHAP explainability
                - Interactive Gradio app
                - Mini dashboard for user health summary

                ## Important Disclaimer

                This app is for educational and portfolio purposes only. It is not a medical diagnostic tool and should not be used for clinical decision-making.
                '''
            )

if __name__ == "__main__":
    demo.launch()
