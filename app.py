import streamlit as st
import pandas as pd
import numpy as np
import joblib

def main():
    st.title("📊 Employee Performance & Retention Predictor")
    st.write("Enter employee details below to predict **Performance Score** and **Retention Probability**.")

    # Load models
    try:
        performance_model = joblib.load("performance_model.pkl")
        retention_model = joblib.load("retention_model.pkl")
    except FileNotFoundError:
        st.error("⚠ Model files not found! Ensure 'performance_model.pkl' and 'retention_model.pkl' exist.")
        return

    # User Inputs
    age = st.number_input("🎂 Age", 18, 65, 30)
    business_travel = st.selectbox("✈ Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    daily_rate = st.slider("📈 Daily Rate", 100, 2000, 800)
    department = st.selectbox("🏢 Department", ["HR", "Finance", "Engineering", "Marketing", "Sales", "IT", "Operations"])
    distance_from_home = st.slider("🚗 Distance from Home (miles)", 1, 50, 10)
    education = st.selectbox("🎓 Education Level", [1, 2, 3, 4, 5])
    education_field = st.selectbox("📚 Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
    environment_satisfaction = st.slider("😊 Environment Satisfaction", 1, 4, 3)
    gender = st.selectbox("👥 Gender", ["Male", "Female"])
    hourly_rate = st.slider("💰 Hourly Rate", 10, 100, 50)
    job_involvement = st.slider("🔨 Job Involvement", 1, 4, 2)
    job_level = st.slider("📊 Job Level", 1, 5, 2)
    job_role = st.selectbox("👔 Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
    job_satisfaction = st.slider("😊 Job Satisfaction", 1, 4, 3)
    marital_status = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced"])
    monthly_income = st.slider("💵 Monthly Income", 1000, 20000, 5000)
    monthly_rate = st.slider("📊 Monthly Rate", 1000, 50000, 10000)
    num_companies_worked = st.slider("🏢 Number of Companies Worked", 0, 10, 3)
    over_time = st.selectbox("⏳ Overtime", ["No", "Yes"])
    percent_salary_hike = st.slider("📈 Percent Salary Hike", 5, 25, 15)
    relationship_satisfaction = st.slider("😊 Relationship Satisfaction", 1, 4, 3)
    stock_option_level = st.slider("📈 Stock Option Level", 0, 3, 1)
    total_working_years = st.slider("📅 Total Working Years", 0, 40, 10)
    training_times_last_year = st.slider("📚 Training Times Last Year", 0, 6, 2)
    work_life_balance = st.slider("⚖ Work-Life Balance", 1, 4, 3)
    years_at_company = st.slider("🏢 Years at Company", 0, 40, 5)
    years_in_current_role = st.slider("🔄 Years in Current Role", 0, 20, 3)
    years_since_last_promotion = st.slider("🏆 Years Since Last Promotion", 0, 15, 2)
    years_with_curr_manager = st.slider("👨‍💼 Years with Current Manager", 0, 20, 4)

    # **🚀 Add Missing Features (Constant Values)**
    employee_count = 1  # Always 1
    over_18 = 1  # Assuming all employees are 18+
    standard_hours = 40  # Default working hours
    employee_number = 9999  # Placeholder, won't affect predictions

    # Encode categorical variables
    business_travel_mapping = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    department_mapping = {"HR": 0, "Finance": 1, "Engineering": 2, "Marketing": 3, "Sales": 4, "IT": 5, "Operations": 6}
    education_field_mapping = {"Life Sciences": 0, "Medical": 1, "Marketing": 2, "Technical Degree": 3, "Human Resources": 4, "Other": 5}
    gender_mapping = {"Male": 0, "Female": 1}
    job_role_mapping = {"Sales Executive": 0, "Research Scientist": 1, "Laboratory Technician": 2, "Manufacturing Director": 3, "Healthcare Representative": 4, "Manager": 5, "Sales Representative": 6, "Research Director": 7, "Human Resources": 8}
    marital_status_mapping = {"Single": 0, "Married": 1, "Divorced": 2}
    over_time_mapping = {"No": 0, "Yes": 1}

    # Convert categorical inputs
    input_data = np.array([
        age, business_travel_mapping[business_travel], daily_rate, department_mapping[department], 
        distance_from_home, education, education_field_mapping[education_field], environment_satisfaction, 
        gender_mapping[gender], hourly_rate, job_involvement, job_level, job_role_mapping[job_role], 
        job_satisfaction, marital_status_mapping[marital_status], monthly_income, monthly_rate, 
        num_companies_worked, over_time_mapping[over_time], percent_salary_hike, relationship_satisfaction, 
        stock_option_level, total_working_years, training_times_last_year, work_life_balance, 
        years_at_company, years_in_current_role, years_since_last_promotion, years_with_curr_manager, 

        # **🚀 Add Missing Features**
        employee_count, over_18, standard_hours, employee_number
    ]).reshape(1, -1)  

    # Predict Performance & Retention
    if st.button("🔮 Predict Performance & Retention"):
        try:
            performance_pred = performance_model.predict(input_data)[0]
            retention_pred = retention_model.predict(input_data)[0]
            retention_prob = retention_model.predict_proba(input_data)[:, 1][0]

            # Display results
            st.success(f"🏆 **Predicted Performance Score:** {round(performance_pred, 2)} / 10")
            if retention_pred == 1:
                st.error(f"⚠ **Retention Status:** Employee is **likely to leave** (Probability: {round(retention_prob * 100, 2)}%)")
            else:
                st.success(f"✅ **Retention Status:** Employee is **likely to stay** (Probability: {round(retention_prob * 100, 2)}%)")
        
        except Exception as e:
            st.error(f"⚠ Error: {e}")

if __name__ == "__main__":
    main()
