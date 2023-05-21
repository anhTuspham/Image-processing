import cv2
import numpy as np
import streamlit as st

# Các hàm xử lý ảnh

def count_connected_components(img):
    ret, temp = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp, 7)
    shape = temp.shape
    M, N = shape[0], shape[1]
    dem = 0
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            if np.array_equal(temp[x, y], [255]):
                mask = np.zeros((M + 2, N + 2), np.uint8)
                cv2.floodFill(temp, mask, (y, x), (color, color, color))
                dem = dem + 1
                color = color + 1
    print('Co %d thanh phan lien thong' % dem)
    a = np.zeros(256, int)
    for x in range(0, M):
        for y in range(0, N):
            r = temp[x, y]
            if np.any(r > 0):
                a[r] = a[r] + 1
    dem = 1
    for r in range(0, 256):
        if a[r] > 0:
            print('%4d   %5d' % (dem, a[r]))
            dem = dem + 1
    return dem

def CountRice(imgin):
    img_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    temp = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, w)
    ret, temp = cv2.threshold(temp, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp = cv2.medianBlur(temp, 3)
    dem, label = cv2.connectedComponents(temp)
    text = 'Co %d hat gao' % (dem - 1)
    # st.write(text)
    a = np.zeros(dem, int)
    M, N = label.shape
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] = a[r] + 1
            if r > 0:
                label[x, y] = label[x, y] + color

    for r in range(0, dem):
        st.write('%4d %10d' % (r, a[r]))

    max_count = a[1]
    rmax = 1
    for r in range(2, dem):
        if a[r] > max_count:
            max_count = a[r]
            rmax = r

    xoa = np.array([], int)
    for r in range(1, dem):
        if a[r] < 0.5 * max_count:
            xoa = np.append(xoa, r)

    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            if r > 0:
                r = r - color
                if r in xoa:
                    label[x, y] = 0
    label = label.astype(np.uint8)
    cv2.putText(label, text, (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return dem



# Hàm chính

def main():
    st.set_page_config(
        page_icon="🖼️",
        page_title="Xử lý ảnh"
    )
    st.subheader('Xử lý ảnh hình thái')

    uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(img, caption='Ảnh gốc', use_column_width=True)

        if st.button('Đếm thành phần liên thông của miếng phi lê gà'):
            num_components = count_connected_components(img)
            st.write(f"Số thành phần liên thông: {num_components}")

        if st.button('Đếm hạt gạo'):
            num_rice = CountRice(img)
            st.write(f"Số hạt gạo: {num_rice}")

if __name__ == '__main__':
    main()
