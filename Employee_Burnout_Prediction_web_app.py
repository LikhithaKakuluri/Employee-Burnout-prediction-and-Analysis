import numpy as np
import pickle
import streamlit as st
import os


#loading the saved model
loaded_model=pickle.load(open("EmployeeBurnOutData_model.sav",'rb'))



img = '''
<style>
.stApp {
    background-image: url("https://images.pexels.com/photos/7135028/pexels-photo-7135028.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2");
    background-size: cover;
    background-position: top center;
    background-repeat: no-repeat;
    background-attachment: local;
    opacity:1;
}
</style>
'''
st.markdown(img, unsafe_allow_html=True)

#creating a function for prediction

def burnout_prediction(input_data):
    
    #changing the input data to numpy array
    input_data_as_numpy_array=np.array(input_data,dtype=float)
    #reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    return prediction



def main():

    

    #giving title
    #st.title("Employee Burnout Prediction Web App")
    st.markdown("<h1 style='text-align: center; color: #C0C0C0;'>Employee Burnout Prediction Web App</h1>", unsafe_allow_html=True)

    #getting the input data from the user
   
#Gender	Company_Type	WFH_Setup_Available	Designation	Resource_Allocation	Mental_Fatigue_Score
#"Gender":{"Male":1,"Female":0},"Company_Type":{"Product":0,"Service":1},"WFH_Setup_Available":{"No":0,"Yes":1}}

    Gender=st.text_input("Enter the Gender(Male:1,Female:0)")

    Company_Type=st.text_input("Enter the Company_Type(Product:0,Service:1)")

    WFH_Setup_Available=st.text_input("Enter Whether Work from home setup available or not(No:0,Yes:1)")

    Designation=st.text_input("Enter the Designation Value (from 0 t0 5)")

    Resource_Allocation=st.text_input("Enter the Resource Allocation value (from 1 to 10)")

    Mental_Fatigue_Score=st.text_input("Enter the Mental Fatigue Score")

    
    #code for prediction

    Report=''

    #creating button for prediction

    if st.button("Get the BurnRate of Employee"):

        Report=burnout_prediction([Gender,Company_Type,WFH_Setup_Available,Designation,Resource_Allocation,Mental_Fatigue_Score])
    
    st.success(Report)




if __name__ =='__main__':
    main()
    
