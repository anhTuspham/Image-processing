import cv2
import numpy as np
import streamlit as st
import random

# Các hàm xử lý ảnh

def add_motion_noise(img, magnitude):
    img_copy = img.copy()
    rows, cols, _ = img_copy.shape
    num_pixels = int(rows * cols * magnitude)
    
    for _ in range(num_pixels):
        x = random.randint(0, cols-1)
        y = random.randint(0, rows-1)
        img_copy[y, x] = (0, 0, 0)
        
    return img_copy

def remove_noise_low(img, kernel_size):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_median = cv2.medianBlur(img_gray, kernel_size)
    return cv2.cvtColor(img_median, cv2.COLOR_GRAY2BGR)

def remove_noise_high(img, kernel_size):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bilateral = cv2.bilateralFilter(img_gray, kernel_size, 75, 75)
    return cv2.cvtColor(img_bilateral, cv2.COLOR_GRAY2BGR)

# Hàm chính

def main():
    st.set_page_config(
        page_icon="🖼️",
        page_title="Xử lý ảnh"
    )
    st.subheader('Khôi phục ảnh')

    uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(img, caption='Ảnh gốc', use_column_width=True)

        if st.button('Add Motion Noise'):
            magnitude = st.slider('Độ lớn của nhiễu', min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            img_noisy = add_motion_noise(img, magnitude)
            st.image(img_noisy, caption='Ảnh sau khi thêm nhiễu chuyển động', use_column_width=True)

        if st.button('Remove Low Noise'):
            kernel_size = st.slider('Kích thước kernel', min_value=3, max_value=25, step=2, value=5)
            img_filtered = remove_noise_low(img, kernel_size)
            st.image(img_filtered, caption='Ảnh sau khi gỡ nhiễu ít', use_column_width=True)

        if st.button('Remove High Noise'):
            kernel_size = st.slider('Kích thước kernel', min_value=3, max_value=25, step=2, value=20)
            img_filtered = remove_noise_high(img, kernel_size)
            st.image(img_filtered, caption='Ảnh sau khi gỡ nhiễu nhiều', use_column_width=True)

if __name__ == '__main__':
    main()
