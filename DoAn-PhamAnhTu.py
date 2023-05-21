import streamlit as st

st.set_page_config(
    page_icon="📷",
    page_title="Image Processing"
)
st.write("# Welcome to MyWebsite! 👋")

st.sidebar.success("Chọn các lựa chọn ở trên.")

st.markdown(
    """
    **👈 Lựa chọn các phương pháp ở cột bên** để hiểu hơn về những gì mà chúng ta có thể xử lý được với những bức ảnh ban đầu
    ### Đinh dạng khuôn mặt trên camera
    - Phát hiện khuôn mặt 
    - Nhận dạng khuôn mặt 
    ### Xử lý ảnh
    - Biến đổi độ sáng và lọc trong không gian
    - Lọc trong miền tần số
    - Khôi phục ảnh
    - Xử lý ảnh hình thái
"""
)