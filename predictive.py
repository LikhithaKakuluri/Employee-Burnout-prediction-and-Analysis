import numpy as np
import pickle

loaded_model=pickle.load(open("C:/Users/kakul/streamlit/Employee/EmployeeBurnOutData_model.sav",'rb'))
input_data=(0,1,0,2.0,3.000000,3.800000)
input_data_as_numpy_array=np.array(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
