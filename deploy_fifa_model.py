import streamlit as st
import joblib
import os
import sklearn

model_filename = 'RandomForestClassifier.pkl'  # Adjust this if your model file name is different
model_filepath = os.path.join(os.path.dirname(__file__), model_filename)

# Load the model
@st.cache(allow_output_mutation=True)
def load_model(filepath):
    return joblib.load(filepath)

model = load_model(model_filepath)


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
