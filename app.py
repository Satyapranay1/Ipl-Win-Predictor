import streamlit as st
import joblib
import pandas as pd

# Custom CSS for advanced styling
st.markdown("""
<style>
    /* Background and main container */
    .stApp {
        background-color: #f5f7fa;
        font-family: 'Arial', sans-serif;
    }

    /* Main title */
    .main-title {
        font-size: 36px;
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
        background: -webkit-linear-gradient(#1a73e8, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }

    /* Subtitles for sections */
    .sub-title {
        font-size: 24px;
        color: #34495e;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Styled select boxes and input fields */
    .stSelectbox, .stNumberInput {
        background-color: #ecf0f1;
        border: none;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        color: #2c3e50;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    /* Button styles */
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #00aaff;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }

    /* Table styling */
    .stTable {
        border-radius: 10px;
        overflow: hidden;
        margin-top: 20px;
        background-color: #ecf0f1;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Header styles */
    .stHeader {
        text-align: center;
        margin-top: 20px;
    }

    .stHeader h1 {
        font-size: 32px;
        color: #34495e;
    }

    .stHeader h2 {
        font-size: 28px;
        color: #1a73e8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">IPL WIN Predictor</p>', unsafe_allow_html=True)

teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 
         'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings', 
         'Rajasthan Royals', 'Delhi Capitals']

city = ['Jaipur', 'Kolkata', 'Chandigarh', 'Ahmedabad', 'Dharamsala',
       'Mumbai', 'New Delhi', 'Dubai', 'Hyderabad', 'Centurion', 'Mohali',
       'Bangalore', 'Chennai', 'Ranchi', 'Cape Town', 'Cuttack',
       'Kimberley', 'London', 'Indore', 'Port Elizabeth', 'Nagpur',
       'Bloemfontein']

pipe = joblib.load(open('model2.pkl','rb'))

with st.expander("Select Teams and Match Conditions"):
    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Select the Batting Team', sorted(teams))
    with col2:
        bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

    s_city = st.selectbox('Select Host City', sorted(city))
    target = st.number_input('Target', min_value=1)

if batting_team == bowling_team:
    st.error("Batting and Bowling Teams cannot be the same")

with st.expander("Match Progress"):
    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input('Score', min_value=0)
    with col4:
        overs = st.number_input('Overs', min_value=0.0, max_value=20.0, step=0.1)
    with col5:
        wickets = st.number_input('Wickets', min_value=0, max_value=10)

# Validation checks
if st.button('Predict Probability'):
    if batting_team == bowling_team:
        st.error("Batting and Bowling Teams cannot be the same.")
    elif overs == 0:
        st.error("Overs cannot be zero.")
    elif score > target:
        st.error("Score cannot exceed the target.")
    elif overs > 20:
        st.error("Overs cannot exceed 20.")
    elif wickets > 10:
        st.error("Wickets cannot exceed 10.")
    else:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        remaining_wickets = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [s_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [remaining_wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        st.subheader("Match Input Data")
        st.table(input_df)

        res = pipe.predict_proba(input_df)
        loss_prob = res[0][0]
        win_prob = res[0][1]

        st.markdown('<p class="sub-title">' + batting_team + " Winning Probability: " + str(round(win_prob * 100, 2)) + "%</p>", unsafe_allow_html=True)
        st.markdown('<p class="sub-title">' + bowling_team + " Winning Probability: " + str(round(loss_prob * 100, 2)) + "%</p>", unsafe_allow_html=True)
