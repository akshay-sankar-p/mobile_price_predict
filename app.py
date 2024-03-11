import streamlit as st
import pickle
import numpy as np
from PIL import Image

def main():
    st.title('Mobile Price Range Prediction')

    image = Image.open('mobile.jpg')
    st.image(image, use_column_width=True)

    # Input fields for mobile phone features
    st.sidebar.subheader("Input Features:")
    battery_power = st.sidebar.text_input('Battery Power (mAh)', '')
    clock_speed = st.sidebar.text_input('Clock Speed (GHz)', '')
    fc = st.sidebar.text_input('Front Camera (MP)', '')
    int_memory = st.sidebar.text_input('Internal Memory (GB)', '')
    mobile_wt = st.sidebar.text_input('Mobile Weight (g)', '')
    pc = st.sidebar.text_input('Primary Camera (MP)', '')
    px_height = st.sidebar.text_input('Pixel Height', '')
    px_width = st.sidebar.text_input('Pixel Width', '')
    ram = st.sidebar.text_input('RAM (MB)', '')
    sc_h = st.sidebar.number_input('Screen Height (cm)', min_value=0)
    sc_w = st.sidebar.number_input('Screen Width (cm)', min_value=0)
    talk_time = st.sidebar.text_input('Talk Time (hours)', '')

    # Checkbox for optional features
    st.sidebar.subheader("Optional Features:")
    bluetooth = st.sidebar.checkbox('Bluetooth')
    dual_sim = st.sidebar.checkbox('Dual SIM')
    three_g = st.sidebar.checkbox('3G')
    four_g = st.sidebar.checkbox('4G')
    touch_screen = st.sidebar.checkbox('Touch Screen')
    wifi = st.sidebar.checkbox('WiFi')
    m_dep = st.sidebar.slider('Mobile Depth', 0.0, 1.0, 1.0, step=0.1)
    n_cores = st.sidebar.selectbox('Number of Cores', ('1', '2', '3', '4', '5', '6', '7', '8'))

    if st.sidebar.button('Predict Price Range'):
        # Validate input data
        if validate_inputs([battery_power, clock_speed, fc, int_memory, mobile_wt, pc, px_height, px_width, ram, talk_time]):
            # Calculate screen size
            # sc_size = np.sqrt((sc_h ** 2) + (sc_w ** 2)) / 2.54
            # Convert optional features to binary
            features = [float(battery_power), int(bluetooth), float(clock_speed), int(dual_sim), float(fc),
                        int(four_g), float(int_memory), float(m_dep), float(mobile_wt), int(n_cores),
                        float(pc), float(px_height), float(px_width), float(ram), float(sc_h), float(sc_w), float(talk_time),
                        int(three_g), int(touch_screen), int(wifi)]

            # Load model and scaler
            model = pickle.load(open('model_svc.sav', 'rb'))
            scaler = pickle.load(open('scaler.sav', 'rb'))

            # Predict price range
            prediction = model.predict(scaler.transform([features]))[0]

            # Display predicted price range
            if prediction == 0:
                st.success('Low Cost')
            elif prediction == 1:
                st.success('Medium Cost')
            elif prediction == 2:
                st.success('High Cost')
            else:
                st.success('Very High Cost')
        else:
            st.error('Please enter valid numerical values for all input features.')


def validate_inputs(inputs):
    for i in inputs:
        if i == '':
            return False
    return True


main()
