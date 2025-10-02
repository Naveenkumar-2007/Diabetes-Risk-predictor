import os
import pickle
import pandas as pd
from crewai import Agent, LLM
from crewai.tools import tool  # ✅ decorator for custom tools
from dotenv import load_dotenv

load_dotenv()

# Load model & scaler
with open('model.pkl',"rb") as f:
    model=pickle.load(f)


# Setup LLM
llm = LLM(
    model=os.getenv("DEFAULT_LLM"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=800  # Reduced to help with rate limits
)

# Define tool
@tool("Diabetes Predictor")
def predict_diabetes(data: dict) -> str:
    """Predict diabetes using the trained ML model."""
    df = pd.DataFrame([data], columns=[
        'Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age'
    ])

    pred = model.predict(df)[0]
    return "Diabetic" if pred == 1 else "Not Diabetic"

# Create a function for direct testing (not decorated)
def test_diabetes_prediction(data: dict) -> str:
    """Direct function for testing diabetes prediction."""
    df = pd.DataFrame([data], columns=[
        'Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age'
    ])

    pred = model.predict(df)[0]
    return "Diabetic" if pred == 1 else "Not Diabetic"

# Create agent
diabetes_agent = Agent(
    role="AI Diabetes Risk Analyst",
    goal="Provide personalized diabetes risk assessment with evidence-based medical recommendations using advanced AI analysis.",
    backstory="""You are an advanced AI medical assistant specialized in diabetes risk assessment. You analyze patient data using machine learning models and provide personalized health recommendations based on clinical guidelines and medical best practices. You always provide clear, actionable advice tailored to each patient's specific health profile.""",
    llm=llm,
    tools=[predict_diabetes],
    verbose=False  # Reduce verbose output
)

# Test the agent setup
if __name__ == "__main__":
    print("✅ Diabetes Agent setup completed successfully!")
    print(f"Agent role: {diabetes_agent.role}")
    print(f"Available tools: {[tool.name for tool in diabetes_agent.tools]}")
    
    # Test the prediction tool directly
    try:
        test_data = {
            'Pregnancies': 2,
            'Glucose': 138,
            'BloodPressure': 62,
            'SkinThickness': 35,
            'Insulin': 0,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.127,
            'Age': 47
        }
        result = test_diabetes_prediction(test_data)
        print(f"✅ Test prediction successful: {result}")
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        
    print("Diabetes agent is ready to use!")
