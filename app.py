# creating the script

# importing required libraries
import pickle
import streamlit as st

from model import Model  # Adjust this based on your actual class and file

# loading the trained model
pickle_in = open('logistic_clf_new2.pkl', 'rb') 
classifier = pickle.load(pickle_in)

# this is the main function in which we define our app  
def main():       
    # header of the page 

    html_temp = """ 
    <div style ="background-color:yellow;padding:13px;border-radius:10px"> 
    <h1 style ="color:black;text-align:center;">Check your Student Dropout Prediction</h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    
    # following lines create boxes in which user can enter data required to make prediction 
    Gender = st.selectbox('Gender',("Male","Female"))

    
    mothers_qualification=st.selectbox("mothers_qualification",(
    "Secondary Education - 12th Year of Schooling or Eq.",
    "Higher Education - Bachelor's Degree",
    "Higher Education - Degree",
    "Higher Education - Master's",
    "Higher Education - Doctorate",
    "Frequency of Higher Education",
    "12th Year of Schooling - Not Completed",
    "11th Year of Schooling - Not Completed",
    "7th Year (Old)",
    "Other - 11th Year of Schooling",
    "10th Year of Schooling",
    "General commerce course",
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    "Technical-professional course",
    "7th year of schooling",
    "2nd cycle of the general high school course",
    "9th Year of Schooling - Not Completed",
    "8th year of schooling",
    "Unknown",
    "Can't read or write",
    "Can read without having a 4th year of schooling",
    "Basic education 1st cycle (4th/5th year) or equiv.",
    "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    "Technological specialization course",
    "Higher education - degree (1st cycle)",
    "Specialized higher studies course",
    "Professional higher technical course",
    "Higher Education - Master (2nd cycle)",
    "Higher Education - Doctorate (3rd cycle)"))

   

    Debtor=st.selectbox('Debtor',("Yes","No"))

    tuition_fees_up_to_date=st.selectbox('tuition_fees_up_to_date',("Yes","No"))
    scholarship_holder=st.selectbox('Scholarship holder',("Yes","No"))

    fathers_qualification= st.selectbox('fathers_qualification', ("Secondary Education - 12th Year of Schooling or Eq.",
    "Higher Education - Bachelor's Degree",
    "Higher Education - Degree",
    "Higher Education - Master's",
    "Higher Education - Doctorate",
    "Frequency of Higher Education",
    "12th Year of Schooling - Not Completed",
    "11th Year of Schooling - Not Completed",
    "7th Year (Old)",
    "Other - 11th Year of Schooling",
    "2nd year complementary high school course",
    "10th Year of Schooling",
    "General commerce course",
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    "Complementary High School Course",
    "Technical-professional course",
    "Complementary High School Course - not concluded",
    "7th year of schooling",
    "2nd cycle of the general high school course",
    "9th Year of Schooling - Not Completed",
    "8th year of schooling",
    "General Course of Administration and Commerce",
    "Supplementary Accounting and Administration",
    "Unknown",
    "Can't read or write",
    "Can read without having a 4th year of schooling",
    "Basic education 1st cycle (4th/5th year) or equiv.",
    "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    "Technological specialization course",
    "Higher education - degree (1st cycle)",
    "Specialized higher studies course",
    "Professional higher technical course",
    "Higher Education - Master (2nd cycle)",
    "Higher Education - Doctorate (3rd cycle)"))



    
    Course=st.selectbox('Course',("Biofuel Production Technologies",
    "Animation and Multimedia Design",
    "Social Service (evening attendance)",
    "Agronomy",
    "Communication Design",
    "Veterinary Nursing",
    "Informatics Engineering",
    "Equinculture",
    "Management",
    "Social Service",
    "Tourism",
    "Nursing",
    "Oral Hygiene",
    "Advertising and Marketing Management",
    "Journalism and Communication",
    "Basic Education",
    "Management (evening attendance)"))



    curricular_units_1st_sem_approved = st.number_input(
    label="Curricular Units 1st Sem (Approved)",
    min_value=0,
    max_value=26,
    step=1)


    curricular_units_2nd_sem_approved = st.number_input(
    label="Curricular Units 2nd Sem (Approved)",
    min_value=0,
    max_value=20,
    step=1)

    curricular_units_1st_sem_grade = st.number_input(
    label="Curricular Units 1st Sem (Grade)",
    min_value=0.0,
    max_value=20.0,
    step=0.5)


    curricular_units_2nd_sem_grade = st.number_input(
    label="Curricular Units 2nd Sem (Grade)",
    min_value=0.0,
    max_value=20.0,
    step=0.5)

    # Curricular Units 1st Sem (Enrolled)
    curricular_units_1st_sem_enrolled = st.number_input(
    label="Curricular Units 1st Sem (Enrolled)",
    min_value=0,
    max_value=26,
    step=1)

    
    curricular_units_2nd_sem_enrolled = st.number_input(
    label="Curricular Units 2nd Sem (Enrolled)",
    min_value=0,
    max_value=23,
    step=1)

    curricular_units_1st_sem_evaluations = st.number_input(
    label="Curricular Units 1st Sem (Evaluations)",
    min_value=0,
    max_value=45,
    step=1)

    curricular_units_2nd_sem_evaluations = st.number_input(
    label="Curricular Units 2nd Sem (Evaluations)",
    min_value=0,
    max_value=33,
    step=1)

    Age_Normalized = st.number_input(
    label="Age_Normalized",
    step=0.1)

    st.markdown('</div>', unsafe_allow_html=True)

    # Clear floats
    st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

    result =""

    # when 'Check' is clicked, make the prediction and store it 
    if st.button("Check"): 
        result = prediction(fathers_qualification, mothers_qualification, curricular_units_1st_sem_approved,
                curricular_units_2nd_sem_approved, curricular_units_2nd_sem_grade,
                curricular_units_1st_sem_grade, curricular_units_2nd_sem_evaluations,
                tuition_fees_up_to_date, curricular_units_1st_sem_evaluations,
                Age_Normalized, curricular_units_1st_sem_enrolled, scholarship_holder,
                Debtor, curricular_units_2nd_sem_enrolled, Course, Gender)
        
 
# defining the function which will make the prediction using the data which the user inputs 
def prediction(fathers_qualification, mothers_qualification, curricular_units_1st_sem_approved,
                curricular_units_2nd_sem_approved, get_curricular_units_2nd_sem_grade,
                curricular_units_1st_sem_grade, curricular_units_2nd_sem_evaluations,
                tuition_fees_up_to_date, get_curricular_units_1st_sem_evaluations,
                Age_Normalized, get_curricular_units_1st_sem_enrolled, scholarship_holder,
                Debtor, get_curricular_units_2nd_sem_enrolled, Course, Gender): 

    # 2. Loading and Pre-processing the data 

    if Gender == "Male":
        Gender = 1
    else:
        Gender = 0

    
    if Debtor == "Yes":
        Debtor = 1
    else:
        Debtor = 0    



    if scholarship_holder == "Yes":
        scholarship_holder = 1
    else:
        scholarship_holder = 0 

    if tuition_fees_up_to_date == "Yes":
        tuition_fees_up_to_date = 1
    else:
        tuition_fees_up_to_date = 0       

    

    if Course == "Biofuel Production Technologies":
        Course=33
    elif Course == "Animation and Multimedia Design":
        Course=171
    elif Course == "Social Service (evening attendance)":
        Course=8014
    elif Course == "Agronomy":
        Course=9003
    elif Course == "Communication Design":
        Course= 9070
    elif Course == "Veterinary Nursing":
        Course= 9085
    elif Course == "Informatics Engineering":
        Course= 9119
    elif Course == "Equinculture":
        Course= 9130
    elif Course == "Management":
        Course= 9147
    elif Course == "Social Service":
        Course= 9238
    elif Course == "Tourism":
        Course= 9254
    elif Course == "Nursing":
        Course= 9500
    elif Course == "Oral Hygiene":
        Course= 9556
    elif Course == "Advertising and Marketing Management":
        Course= 9670
    elif Course == "Journalism and Communication":
        Course= 9773
    elif Course == "Basic Education":
        Course= 9853
    elif Course == "Management (evening attendance)":
        Course= 9991
    

   

    if mothers_qualification == "Secondary Education - 12th Year of Schooling or Eq.":
        mothers_qualification = 1
    elif mothers_qualification == "Higher Education - Bachelor's Degree":
        mothers_qualification = 2
    elif mothers_qualification == "Higher Education - Degree":
        mothers_qualification = 3
    elif mothers_qualification == "Higher Education - Master's":
        mothers_qualification = 4
    elif mothers_qualification == "Higher Education - Doctorate":
        mothers_qualification = 5
    elif mothers_qualification == "Frequency of Higher Education":
        mothers_qualification = 6
    elif mothers_qualification == "12th Year of Schooling - Not Completed":
        mothers_qualification = 9
    elif mothers_qualification == "11th Year of Schooling - Not Completed":
        mothers_qualification = 10
    elif mothers_qualification == "7th Year (Old)":
        mothers_qualification = 11
    elif mothers_qualification == "Other - 11th Year of Schooling":
        mothers_qualification = 12
    elif mothers_qualification == "10th Year of Schooling":
        mothers_qualification = 14
    elif mothers_qualification == "General commerce course":
        mothers_qualification = 18
    elif mothers_qualification == "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.":
        mothers_qualification = 19
    elif mothers_qualification == "Technical-professional course":
        mothers_qualification = 22
    elif mothers_qualification == "7th year of schooling":
        mothers_qualification = 26
    elif mothers_qualification == "2nd cycle of the general high school course":
        mothers_qualification = 27
    elif mothers_qualification == "9th Year of Schooling - Not Completed":
        mothers_qualification = 29
    elif mothers_qualification == "8th year of schooling":
        mothers_qualification = 30
    elif mothers_qualification == "Unknown":
        mothers_qualification = 34
    elif mothers_qualification == "Can't read or write":
        mothers_qualification = 35
    elif mothers_qualification == "Can read without having a 4th year of schooling":
        mothers_qualification = 36
    elif mothers_qualification == "Basic education 1st cycle (4th/5th year) or equiv.":
        mothers_qualification = 37
    elif mothers_qualification == "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.":
        mothers_qualification = 38
    elif mothers_qualification == "Technological specialization course":
        mothers_qualification = 39
    elif mothers_qualification == "Higher education - degree (1st cycle)":
        mothers_qualification = 40
    elif mothers_qualification == "Specialized higher studies course":
        mothers_qualification = 41
    elif mothers_qualification == "Professional higher technical course":
        mothers_qualification = 42
    elif mothers_qualification == "Higher Education - Master (2nd cycle)":
        mothers_qualification = 43
    elif mothers_qualification == "Higher Education - Doctorate (3rd cycle)":
        mothers_qualification = 44
   

    if fathers_qualification == "Secondary Education - 12th Year of Schooling or Eq.":
        fathers_qualification = 1
    elif fathers_qualification == "Higher Education - Bachelor's Degree":
        fathers_qualification = 2
    elif fathers_qualification == "Higher Education - Degree":
        fathers_qualification = 3
    elif fathers_qualification == "Higher Education - Master's":
        fathers_qualification = 4
    elif fathers_qualification == "Higher Education - Doctorate":
        fathers_qualification =5
    elif fathers_qualification == "Frequency of Higher Education":
        fathers_qualification = 6
    elif fathers_qualification == "12th Year of Schooling - Not Completed":
        fathers_qualification = 9
    elif fathers_qualification == "11th Year of Schooling - Not Completed":
        fathers_qualification = 10
    elif fathers_qualification == "7th Year (Old)":
        fathers_qualification = 11
    elif fathers_qualification == "Other - 11th Year of Schooling":
        fathers_qualification = 12
    elif fathers_qualification == "2nd year complementary high school course":
        fathers_qualification = 13
    elif fathers_qualification == "10th Year of Schooling":
        fathers_qualification = 14
    elif fathers_qualification == "General commerce course":
        fathers_qualification = 18
    elif fathers_qualification == "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.":
        fathers_qualification = 19
    elif fathers_qualification == "Complementary High School Course":
        fathers_qualification = 20
    elif fathers_qualification == "Technical-professional course":
        fathers_qualification = 22
    elif fathers_qualification == "Complementary High School Course - not concluded":
        fathers_qualification = 25
    elif fathers_qualification == "7th year of schooling":
        fathers_qualification = 26
    elif fathers_qualification == "2nd cycle of the general high school course":
        fathers_qualification = 27
    elif fathers_qualification == "9th Year of Schooling - Not Completed":
        fathers_qualification = 29
    elif fathers_qualification == "8th year of schooling":
        fathers_qualification = 30
    elif fathers_qualification == "General Course of Administration and Commerce":
        fathers_qualification = 31
    elif fathers_qualification == "Supplementary Accounting and Administration":
        fathers_qualification = 33
    elif fathers_qualification == "Unknown":
        fathers_qualification = 34
    elif fathers_qualification == "Can't read or write":
        fathers_qualification = 35
    elif fathers_qualification == "Can read without having a 4th year of schooling":
        fathers_qualification = 36
    elif fathers_qualification == "Basic education 1st cycle (4th/5th year) or equiv.":
        fathers_qualification = 37
    elif fathers_qualification == "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.":
        fathers_qualification = 38
    elif fathers_qualification == "Technological specialization course":
        fathers_qualification = 39
    elif fathers_qualification == "Higher education - degree (1st cycle)":
        fathers_qualification = 40
    elif fathers_qualification == "Specialized higher studies course":
        fathers_qualification = 41
    elif fathers_qualification == "Professional higher technical course":
        fathers_qualification = 42
    elif fathers_qualification == "Higher Education - Master (2nd cycle)":
        fathers_qualification = 43
    elif fathers_qualification == "Higher Education - Doctorate (3rd cycle)":
        fathers_qualification = 44



    
    if 0 <= curricular_units_1st_sem_approved <= 26:
        curricular_units_1st_sem_approved = curricular_units_1st_sem_approved

        

    if 0 <= curricular_units_2nd_sem_approved <= 20:
        curricular_units_2nd_sem_approved= curricular_units_2nd_sem_approved
    

    if 0.0 <= curricular_units_1st_sem_grade <= 20.0:
        curricular_units_1st_sem_grade= curricular_units_1st_sem_grade
    

    if 0.0 <= get_curricular_units_2nd_sem_grade <= 20.0:
        get_curricular_units_2nd_sem_grade= get_curricular_units_2nd_sem_grade
    

    if 0 <= get_curricular_units_1st_sem_enrolled <= 26:
        get_curricular_units_1st_sem_enrolled= get_curricular_units_1st_sem_enrolled
    

    if 0 <= get_curricular_units_2nd_sem_enrolled <= 23:
        get_curricular_units_2nd_sem_enrolled= get_curricular_units_2nd_sem_enrolled
    

    if 0 <= get_curricular_units_1st_sem_evaluations <= 45:
        get_curricular_units_1st_sem_evaluations= get_curricular_units_1st_sem_evaluations
    

    if 0 <= curricular_units_2nd_sem_evaluations <= 33:
        curricular_units_2nd_sem_evaluations= curricular_units_2nd_sem_evaluations

    
    if 0<=Age_Normalized<=100:
        Age_Normalized= Age_Normalized

    
    # prediction = classifier.predict( 
    #     [[fathers_qualification, mothers_qualification, curricular_units_1st_sem_approved,
    #             curricular_units_2nd_sem_approved, get_curricular_units_2nd_sem_grade,
    #             curricular_units_1st_sem_grade, curricular_units_2nd_sem_evaluations,
    #             tuition_fees_up_to_date, get_curricular_units_1st_sem_evaluations,
    #             Age_Normalized, get_curricular_units_1st_sem_enrolled, scholarship_holder,
    #             debtor, get_curricular_units_2nd_sem_enrolled, Course, Gender]])
    
    import pandas as pd

    data = pd.DataFrame({
        'fathers_qualification': [fathers_qualification],
        'mothers_qualification': [mothers_qualification],
        'curricular_units_1st_sem_approved': [curricular_units_1st_sem_approved],
        'curricular_units_2nd_sem_approved': [curricular_units_2nd_sem_approved],
        'get_curricular_units_2nd_sem_grade': [get_curricular_units_2nd_sem_grade],
        'curricular_units_1st_sem_grade': [curricular_units_1st_sem_grade],
        'curricular_units_2nd_sem_evaluations': [curricular_units_2nd_sem_evaluations],
        'tuition_fees_up_to_date': [tuition_fees_up_to_date],
        'get_curricular_units_1st_sem_evaluations': [get_curricular_units_1st_sem_evaluations],
        'Age_Normalized': [Age_Normalized],
        'get_curricular_units_1st_sem_enrolled': [get_curricular_units_1st_sem_enrolled],
        'scholarship_holder': [scholarship_holder],
        'Debtor': [Debtor],
        'get_curricular_units_2nd_sem_enrolled': [get_curricular_units_2nd_sem_enrolled],
        'Course': [Course],
        'Gender': [Gender]
    })

    prediction = classifier.predict(data)

    st.success('The Student will: {}'.format(prediction))

    if prediction=="Graduate":
        st.success("The Student is on right track and in most likely circumstances he/she will graduate.")
    elif prediction=="Dropout":
        st.success("There are chances the student will not be able to graduate and he/she can look at alternate career options or put in extra effort to get on right track..")

    
     
if __name__ == '__main__':
    main()