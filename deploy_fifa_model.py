import streamlit as st
import pandas as pd
import pickle

file_path = 'C:/Users/user/OneDrive - Ashesi University/Sophomore Year Semester II/Introduction to Artificial Intelligence/Fifa_Assignment/RandomForestRegressor.pkl'


with open(file_path, 'rb') as file:
    model = pickle.load(file)
# try:
#     with open(file_path, 'rb') as file:
#         model = pickle.load(file)
#         print("Model loaded successfully!")
#     except FileNotFoundError:
#         print(f"Error: File '{file_path}' not found.")
#     except Exception as e:
#         print(f"Error loading model: {e}")



def main(): 

    st.title("Player Rating Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Player Rating Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Input fields for features
    potential = st.number_input("Potential", value=0.0)
    movement_reactions = st.number_input("Movement Reactions", value=0.0)
    value_in_euros = st.number_input("Value in Euros", value=0.0)
    wage_in_euros = st.number_input("Wage in Euros", value=0.0)
    passing = st.number_input("Passing Rate", value=0.0)
    dribbling = st.number_input("Dribbling", value=0.0)
    physic = st.number_input("Physic", value=0.0)
    mentality_composure = st.number_input("Mentality Composure", value=0.0)
    
    # Prediction button
    if st.button("Predict"): 
        # Create a DataFrame from user inputs
        features = [[potential, movement_reactions, value_in_euros, wage_in_euros, passing, dribbling, physic, mentality_composure]]
        df = pd.DataFrame(features, columns=['Potential', 'Movement Reactions', 'Value in Euros', 'Wage in Euros', 'Passing', 'Dribbling', 'Physic', 'Mentality Composure'])
        
        # Convert DataFrame to list of lists
        features_list = df.values.tolist()
        
        # Make prediction using the loaded model
        prediction = model.predict(features_list)
        
        # Display prediction
        st.write(f'Player Overall Rating: {prediction[0]}')
      
if __name__=='__main__': 
    main()
