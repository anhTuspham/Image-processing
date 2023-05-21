import cv2
import numpy as np
import streamlit as st

def negative(img):
    return 255 - img

def histogram(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    return hist

def hist_equal(img):
    equ = cv2.equalizeHist(img)
    return equ

def hist_equal_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def local_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def hist_stat(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    hist_equalized = np.interp(img.flatten(), range(256), cdf_normalized).reshape(img.shape)
    hist_equalized = hist_equalized.astype(np.uint8)

    return hist_equalized

def my_box_filter(img):
    kernel = np.ones((3, 3), np.float32) / 9
    return cv2.filter2D(img, -1, kernel)

def median_filter(img):
    return cv2.medianBlur(img, 7)

def sharpen(img):
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def gradient(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(sobelx, sobely).astype(np.uint8)

def main():
    st.set_page_config(
        page_icon="🙂",
        page_title="Xử lý hình ảnh"
    )
    
    st.subheader('Biến đổi độ sáng và lọc trong không gian')

    uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        st.image(img, caption='Ảnh gốc', use_column_width=True)

        if st.button('Negative'):
            img_out = negative(img)
            st.image(img_out, caption='Ảnh Negative', use_column_width=True)

        if st.button('Histogram'):
            hist = histogram(img)
            st.bar_chart(hist)

        if st.button('Histogram Equalization'):
            img_out = hist_equal(img)
            st.image(img_out, caption='Ảnh Equalization', use_column_width=True)

        if st.button('Histogram Equalization (Color)'):
            img_out = hist_equal_color(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            st.image(img_out, caption='Ảnh Equalization (Color)', use_column_width=True)

        if st.button('Local Histogram Equalization'):
            img_out = local_hist(img)
            st.image(img_out, caption='Ảnh Local Equalization', use_column_width=True)

        if st.button('Histogram Statistics Equalization'):
            img_out = hist_stat(img)
            st.image(img_out, caption='Ảnh Equalization (Stat)', use_column_width=True)

        if st.button('My Box Filter'):
            img_out = my_box_filter(img)
            st.image(img_out, caption='Ảnh Box Filter', use_column_width=True)

        if st.button('Median Filter'):
            img_out = median_filter(img)
            st.image(img_out, caption='Ảnh Median Filter', use_column_width=True)

        if st.button('Sharpen'):
            img_out = sharpen(img)
            st.image(img_out, caption='Ảnh Sharpen', use_column_width=True)

        if st.button('Gradient'):
            img_out = gradient(img)
            st.image(img_out, caption='Ảnh Gradient', use_column_width=True)

if __name__ == '__main__':
    main()
