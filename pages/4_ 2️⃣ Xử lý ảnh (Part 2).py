import cv2
import numpy as np
import streamlit as st

def spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    spectrum = 20 * np.log(np.abs(fshift))
    spectrum = np.uint8(spectrum)
    return spectrum
def highpass_filter(img, cutoff_freq):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_float = np.float32(img_gray) / 255.0
    img_dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    shift_dft = np.fft.fftshift(img_dft)

    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    shift_dft[crow - cutoff_freq: crow + cutoff_freq, ccol - cutoff_freq: ccol + cutoff_freq] = 0

    shift_idft = np.fft.ifftshift(shift_dft)
    img_idft = cv2.idft(shift_idft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    img_filtered = np.clip(img_idft * 255.0, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)
def draw_notch_reject_filter(img, center, radius):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img_gray.shape
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    mask = 255 - mask
    img_filtered = cv2.bitwise_and(img_gray, mask)
    return cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)
def remove_moire(img, kernel_size):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_median = cv2.medianBlur(img_gray, kernel_size)
    return cv2.cvtColor(img_median, cv2.COLOR_GRAY2BGR)

def main():
    st.set_page_config(
        page_icon="🖼️",
        page_title="Xử lý ảnh"
    )
    st.subheader('Lọc trong miền tần số')

    uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(img, caption='Ảnh gốc', use_column_width=True)

        if st.button('Spectrum'):
            img_spectrum = spectrum(img)
            st.image(img_spectrum, caption='Phổ tần số', use_column_width=True)
            
        if st.button('Highpass Filter'):
            cutoff_freq = st.slider('Tần số cắt', min_value=10, max_value=min(img.shape[0], img.shape[1]) // 2, step=10, value=20)
            img_filtered = highpass_filter(img, cutoff_freq)
            st.image(img_filtered, caption='Ảnh sau khi áp dụng highpass filter', use_column_width=True)

        if st.button('Draw Notch Reject Filter'):
            center_x = st.slider('Tọa độ x của tâm', min_value=0, max_value=img.shape[1], step=1, value=img.shape[1] // 2)
            center_y = st.slider('Tọa độ y của tâm', min_value=0, max_value=img.shape[0], step=1, value=img.shape[0] // 2)
            radius = st.slider('Bán kính', min_value=1, max_value=min(img.shape[0], img.shape[1]) // 2, step=1, value=min(img.shape[0], img.shape[1]) // 4)
            img_filtered = draw_notch_reject_filter(img, (center_x, center_y), radius)
            st.image(img_filtered, caption='Ảnh sau khi vẽ notch reject filter', use_column_width=True)

        if st.button('Remove Moire'):
            kernel_size = st.slider('Kích thước kernel', min_value=3, max_value=25, step=2, value=5)
            img_filtered = remove_moire(img, kernel_size)
            st.image(img_filtered, caption='Ảnh sau khi xử lý moire', use_column_width=True)

if __name__ == '__main__':
    main()
