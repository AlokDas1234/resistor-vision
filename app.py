import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import base64
import os
import asyncio

asyncio.set_event_loop(asyncio.new_event_loop())

# Define color code dictionaries
color_codes = {
    "black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
    "green": 5, "blue": 6, "violet": 7, "gray": 8, "white": 9
}

multipliers = {
    "black": 1, "brown": 10, "red": 100, "orange": 1_000, "yellow": 10_000,
    "green": 100_000, "blue": 1_000_000, "violet": 10_000_000, "gray": 100_000_000, "white": 1_000_000_000,
    "gold": 0.1, "silver": 0.01
}

tolerances = {
    "brown": "±1%", "red": "±2%", "green": "±0.5%", "blue": "±0.25%",
    "violet": "±0.1%", "gray": "±0.05%", "gold": "±5%", "silver": "±10%"
}

# Load model once
# @st.cache_resource
def load_model():
    return YOLO("model/best (9).pt")

model = load_model()

# Functions

def correct_orientation(detected_colors):
    if not detected_colors:
        return []
    if detected_colors[0] in ["gold", "silver"]:
        detected_colors.reverse()
    return detected_colors

def sort_band(results):
    bands = []
    for box in results[0].boxes:
        color_name = model.names[int(box.cls)]
        x_pos = box.xyxy[0][0].item()
        y_pos = box.xyxy[0][1].item()
        bands.append((color_name, x_pos, y_pos))

    if not bands:
        return None

    # Sort Left to Right
    bands_lr = sorted(bands, key=lambda x: x[1])
    if bands_lr[0][0] in ["gold", "silver"]:
        return [b[0] for b in bands_lr]

    # Right to Left
    bands_rl = sorted(bands, key=lambda x: x[1], reverse=True)
    if bands_rl[0][0] in ["gold", "silver"]:
        return [b[0] for b in bands_rl]

    # Top to Bottom
    bands_tb = sorted(bands, key=lambda y: y[2])
    if bands_tb[0][0] in ["gold", "silver"]:
        return [b[0] for b in bands_tb]

    # Bottom to Top
    bands_bt = sorted(bands, key=lambda x: x[2], reverse=True)
    if bands_bt[0][0] in ["gold", "silver"]:
        return [b[0] for b in bands_bt]

    return None

def calculate_resistance(detected_colors):
    detected_colors = correct_orientation(detected_colors)
    if not detected_colors or len(detected_colors) < 3:
        return "Invalid resistor reading (not enough bands detected)"

    first_digit = color_codes.get(detected_colors[0])
    second_digit = color_codes.get(detected_colors[1])
    multiplier = multipliers.get(detected_colors[2])

    if None in [first_digit, second_digit, multiplier]:
        return "Invalid resistor color bands."

    resistance = (first_digit * 10 + second_digit) * multiplier
    tolerance = tolerances.get(detected_colors[3], "±20%") if len(detected_colors) > 3 else "±20%"

    return f"Resistance: {resistance}Ω {tolerance}"

# Streamlit UI
st.title("🎨 Resistor Color Code Detector")
st.write("Upload a resistor image to detect color bands and calculate resistance.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image",  use_container_width=True)

    # YOLO detection
    with st.spinner("Detecting color bands..."):
        results = model.predict(source=image, conf=0.5, save=False)

    bands = sort_band(results)

    if bands:
        st.success(f"Detected bands: {bands}")
        resistance_result = calculate_resistance(bands)
        st.info(resistance_result)

        # Annotated result image
        result_array = results[0].plot()
        result_pil = Image.fromarray(result_array[:, :, ::-1])
        st.image(result_pil, caption="Detected Color Bands",  use_container_width=True)
    else:
        st.error("No color bands detected. Please try another image.")
