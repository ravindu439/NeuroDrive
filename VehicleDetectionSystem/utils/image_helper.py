

import base64
from pathlib import Path
import streamlit as st


def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None



def display_sidebar_logo(logo_path, width=150):

    logo_b64 = get_base64_image(logo_path)
    
    if logo_b64:
        st.sidebar.markdown(f"""
        <div class="sidebar-logo">
            <img src="data:image/png;base64,{logo_b64}" 
                 width="{width}" 
                 alt="Logo">
        </div>
        """, unsafe_allow_html=True)
