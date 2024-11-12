import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import catboost
import shap
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")

st.title("Stroke Prediction Project")

# Section configuration and corresponding buttons
if 'section' not in st.session_state:
    st.session_state.section = 'Project_description_stroke'

col1, col2 = st.columns(2)

with col1:
    if st.button('Project Description'):
        st.session_state.section = 'Project_description_stroke'

with col2:
    if st.button('Predict'):
        st.session_state.section = 'Predict'

#Read txt files to incorporate in Section Project Description
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

Text1 = read_file('Text/Intro.txt')
Text2 = read_file('Text/Preliminary_Data_Analysis1.txt')
Text3 = read_file('Text/Preliminary_Data_Analysis2.txt')
Text4 = read_file('Text/Preliminary_Data_Analysis3.txt')
Text5 = read_file('Text/Preliminary_Data_Analysis4.txt')
Text6 = read_file('Text/Preliminary_Data_Analysis5.txt')
Text7 = read_file('Text/Data_Preprocessing.txt')
Text8 = read_file('Text/Training.txt')
Text9 = read_file('Text/Results1.txt')
Text10 = read_file('Text/Test.txt')
Text11 = read_file('Text/Conclusions.txt')


if st.session_state.section == 'Plan':
    st.header("Estructura de la página")
    
    st.markdown("""#### Introducción""")

elif st.session_state.section == 'Project_description_stroke':
    st.header("Project description")
    st.markdown(Text1)
    st.subheader("Preliminary data analysis")
    st.markdown(Text2)
    st.image('Images/Age-Stroke.png')  
    st.markdown(Text3)
    st.image('Images/Glucose-Stroke.png')
    st.markdown(Text4)  
    st.image('Images/Age-Glucose.png')
    st.markdown(Text5)  
    st.image('Images/Cramers-V.png')
    st.markdown(Text6)
    st.subheader("Data preprocessing")
    st.markdown(Text7)
    st.subheader("Prediction model training")
    st.markdown(Text8)
    st.subheader("Training results and prediction test")
    st.markdown(Text9)
    st.markdown(Text10)
    st.subheader("Conclusions and further steps for improving prediction performance")
    st.markdown(Text11)

elif st.session_state.section == 'Predict':
    st.title("Prediction Section")
    st.write("""
        Welcome to the prediction section. Here, you will be able to execute the trained model (please see section "Project Description") on a set of data to see
        if it predicts a positive stroke occurrence.
        Please use the interface below to introduce the data.
    """)
    
    if 'Age' not in st.session_state:
        st.session_state["Age"] = np.nan
    if 'Glucose' not in st.session_state:
        st.session_state["Glucose"] = np.nan
    if 'Smoking_status' not in st.session_state:
        st.session_state["Smoking_status"] = None
    if 'Sex' not in st.session_state:
        st.session_state["Sex"] = None
    if 'Hypertension' not in st.session_state:
        st.session_state["Hypertension"] = False
    if 'Work_type' not in st.session_state:
        st.session_state["Work_type"] = None
    if 'Ever_married' not in st.session_state:
        st.session_state["Ever_married"] = False
    if "Heart_disease" not in st.session_state:
        st.session_state["Heart_disease"] = False
    if "Residence_type" not in st.session_state:
        st.session_state["Residence_type"] = None
    if "BMI" not in st.session_state:
        st.session_state["BMI"] = np.nan

    col1, col2 = st.columns(2)
    with col1:
        st.radio("Select the sex of the subject", ("Male", "Female"), key="Sex")
        st.number_input("Load the subject's age", min_value=0.0, key="Age")
        st.number_input("Load the subject's average glucose level", min_value=0.0, key="Glucose")
        st.write("Please use the checkboxes to indicate wether the following apply:")
        col3, col4, col5 = st.columns(3)
        with col3:
            st.checkbox("Hypertension", key="Hypertension")
        with col4:
            st.checkbox("Heart disease", key="Heart_disease")
        with col5:
            st.checkbox("Is or has been married", key="Ever_married")
        st.selectbox("Select the work type that best describes the subject's working history", 
                     ["Private", "Self-employed", "Government job", "Never worked", "Children"], key="Work_type")
        st.radio("Select the subject's residence type", ("Urban", "Rural"), key="Residence_type")
        st.number_input("Load the subject's BMI", min_value=0.0, key="BMI")
        st.selectbox("Select the subject's smoking status", ["Never smoked", "Formerly smoked", "Smokes", "Unknown"], key="Smoking_status")

        col6, col7 = st.columns(2)
        with col6:
            if st.button("Clear data", key="Clear data"):
                st.session_state.clear()
                st.session_state.section = 'Predict'
                st.rerun()
                
        with col7:
            def Predict_callback():
                st.session_state['Predict'] = 1
            if 'Predict' not in st.session_state:
                st.session_state['Predict'] = 0

            st.button("Predict", key="Prediction", on_click=Predict_callback)
            
    with col2:
        def Data_loading_func():
             df = pd.DataFrame ({'Sex' : ["Male", "Female", "Female", "Female", "Female", st.session_state["Sex"]],
                               'Age' : [25, 30, 35, 40, 45, st.session_state["Age"]],
                               'Glucose' : [80, 82, 84, 86, 88, st.session_state["Glucose"]],
                               'Hypertension' : [False, False, False, False, True, st.session_state["Hypertension"]],
                               'Heart_disease' : [False, False, False, False, True, st.session_state["Heart_disease"]],
                               'Ever_married' : [False, False, False, False, True, st.session_state["Ever_married"]],
                               'Work_type' : ["Private", "Self-employed", "Government job", "Never worked", "Children", st.session_state["Work_type"]],
                               'Residence_type' : ["Urban", "Urban", "Urban", "Urban", "Rural", st.session_state["Residence_type"]],
                               'BMI' : [18, 20, 22, 24, 26, st.session_state["BMI"]],
                               'Smoking_status' : ["Never smoked", "Smokes", "Formerly smoked", "Unknown", "Smokes", st.session_state["Smoking_status"]]})
             df = df.replace({True: 1, False: 0})
             df['Ever_married'] = df['Ever_married'].replace({1: 'Yes', 0: 'No'})
             return df


        def validation_func(df):
            if st.session_state['Predict'] == 1:
                df_v = df
                if (df_v['Sex'].isna().any() or
                    df_v['Age'].isna().any() or
                    df_v['Age'].iloc[5] > 120 or
                    df_v['Glucose'].isna().any() or
                    ~((df_v['Glucose'].iloc[5] >= 35) & (df_v['Glucose'].iloc[5] <= 350)) or
                    df_v['Work_type'].isna().any() or
                    df_v['Residence_type'].isna().any() or
                    df_v['BMI'].isna().any() or
                    ~((df_v['BMI'].iloc[5] >= 12) & (df_v['BMI'].iloc[5] <= 70)) or
                    df_v['Smoking_status'].isna().any()):
                    st.write ("""You have entered extreme values or missed to inform relevant data for certain variables. Please check
                              the values introduced for the following variables:""")
                    if df_v['Sex'].isna().any():
                        st.write("- Sex: You have not entered a value for this variable.")
                    if df_v['Age'].isna().any():
                        st.write("- Age: You have not entered a value for this variable.")
                    if df_v['Age'].iloc[5] > 120:
                        st.write("- Age: Ages over 120 are rare and considered as the result of a loading error. Please revise.")
                    if df_v['Glucose'].isna().any():
                        st.write("- Glucose: You have not entered a value for this variable.")
                    elif ~((df_v['Glucose'].iloc[5] >= 35) & (df_v['Glucose'].iloc[5] <= 350)):
                        st.write("- Average glucose level: Levels below 35 and over 350 are rare and considered as the result of a loading error.")
                    if df_v['Work_type'].isna().any():
                        st.write("- Work type: You have not entered a value for this variable.")
                    if df_v['Residence_type'].isna().any():
                        st.write("- Residence type: You have not entered a value for this variable.")
                    if df_v['BMI'].isna().any():
                        st.write("- BMI: You have not entered a value for this variable.")
                    elif ~((df_v['BMI'].iloc[5] >= 12) & (df_v['BMI'].iloc[5] <= 70)):
                        st.write("- BMI: Values below 12 and over 70 are rare and considered as the result of a loading error. Please revise.")
                    if df_v['Smoking_status'].isna().any():
                        st.write("- Smoking status: You have not entered a value for this variable.")
                    st.markdown("""**As prediction quality using the trained model may be affected by incomplete data, please fill all missing values or click on the 
                                "Fill missing values" button below to fill them with the values most frequently found in the Stroke dataset.**""")
                    if st.button("Fill missing values", key="Fill_missing_values"):
                        modes = joblib.load('modes.pkl')
                        column_names = df.columns
                        for column in column_names:
                            df_v[column].fillna(modes[column], inplace=True)
            return df_v

                        
        def encoding_func(df_v):
            encoder1=OneHotEncoder(sparse_output=False)
            variables_to_encode = ['Sex', 'Ever_married', 'Work_type', 'Residence_type', 'Smoking_status']
            encoder2=ColumnTransformer(
                transformers=[("cat", encoder1, variables_to_encode)],
                remainder="passthrough")
            X_dummy = encoder2.fit_transform(df_v)
            df_dummy = pd.DataFrame(X_dummy, columns=encoder2.get_feature_names_out())
            cols=list(df_dummy.columns)
            cols=[x.split('__')[-1] for x in cols]
            df_dummy.columns=cols
            df1 = df_dummy.drop(columns=['Sex_Male', 
                                         'Ever_married_No',
                                         'Residence_type_Rural',
                                         'Work_type_Children',
                                         'Smoking_status_Unknown',
                                        ])
            df1 = df1.iloc[5:]
            return df1
          
        def scaling_func(df1):   
            df2 = df1.drop(['Age', 'Glucose', 'BMI'], axis=1)
            df2[['Age', 'Glucose', 'BMI']] = scaler1.transform(df1[['Age', 'Glucose', 'BMI']])
            df2 = df2[[ 'Sex_Female',
                        'Ever_married_Yes',
                        'Work_type_Government job',
                        'Work_type_Never worked',
                        'Work_type_Private',
                        'Work_type_Self-employed',
                        'Residence_type_Urban',
                        'Smoking_status_Formerly smoked',
                        'Smoking_status_Never smoked',
                        'Smoking_status_Smokes',
                        'Age',
                        'Hypertension',
                        'Heart_disease',
                        'Glucose',
                        'BMI',
                        ]]
            return df2
            
            
        def predicting_func(df2):
            try:
                result = best_model1.predict(df2)
            except:
                st.markdown('## An error occurred: the model cannot fit incomplete data')
            return result

        if st.session_state["Predict"] == 1:
            scaler1 = joblib.load('Minmaxscaler.pkl')
            best_model1 = joblib.load('best_model.pkl')
            final_model = best_model1.named_steps['Estimation']
            df = Data_loading_func()
            df_v = validation_func(df)
            df1 = encoding_func(df_v)
            df2 = scaling_func(df1)
            result = predicting_func(df2)
            if result == 1:
                st.markdown("#### Result: The model predicts the occurrence of a stroke based on the loaded data.")
            elif result == 0:
                st.markdown("#### Result: The model does not predict the occurrence of a stroke based on the loaded data.")
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer(df2)
            st.markdown("#### Prediction Explanation: on this graph, you will find the main variables that contributed to this result (positive values contribute to a positive result and negative values contribute to a negative result).")
            shap.bar_plot(shap_values[0].values, feature_names=df2.columns)
            st.pyplot(plt)
