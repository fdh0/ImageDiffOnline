import streamlit as st
from PIL import Image
import numpy as np
import cv2


def toImgPIL(imgOpenCV):
    imgOpenCV_=imgOpenCV
    if(len(imgOpenCV_.shape)==2):
        imgOpenCV_=cv2.cvtColor(imgOpenCV_, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(cv2.cvtColor(imgOpenCV_, cv2.COLOR_BGR2RGB))


st.sidebar.markdown('''
## 在线图像对比 
-- Online Image Difference -''')


uploaded_files = st.sidebar.file_uploader("1 请选择标准图像(select the image 1)", accept_multiple_files=True)
image_place1 = st.empty()

for uploaded_file in uploaded_files:
    print(uploaded_file)
    image = Image.open(uploaded_file)
    with image_place1:
        st.image(image, caption='标准图-model image',use_column_width=True)
        open_cv_image1 = np.array(image)
        if(image.mode=='RGB'):
        # Convert RGB to BGR 
            open_cv_image1 = open_cv_image1[:, :, ::-1].copy() 

        


uploaded_files = st.sidebar.file_uploader("2 请选择检测图像(select image 2)", accept_multiple_files=True)
image_place2 = st.empty()

for uploaded_file in uploaded_files:
    print(uploaded_file)
    image = Image.open(uploaded_file)
    with image_place2:
        st.image(image, caption='检测图- check image',use_column_width=True)
        open_cv_image2 = np.array(image)
        if(image.mode=='RGB'):
        # Convert RGB to BGR 
            open_cv_image2 = open_cv_image2[:, :, ::-1].copy() 


check = st.sidebar.button('3. 开始检测(begin check)',key = "on_checked")

if(check):
    if "on_check" not in st.session_state:
        st.session_state["on_check"] = True
else:
    if "on_check" not in st.session_state:
        st.session_state["on_check"] = False


image_place3 = st.empty()

#if(st.session_state["on_check"] or check):
    # 开始检测
shape1 = open_cv_image1.shape
print(shape1)
shape2 = open_cv_image2.shape
print(shape2)
if shape1 == shape2:
    opencv_diff_img = cv2.absdiff(open_cv_image2,open_cv_image1)
else:
    open_cv_image2 = cv2.resize(open_cv_image2,[shape1[1],shape1[0]])
    print(open_cv_image2.shape)
    opencv_diff_img = cv2.absdiff(open_cv_image2,open_cv_image1)

with image_place3:
    st.image(opencv_diff_img,use_column_width=True,caption="区别图,difference")


st.sidebar.markdown('4. 调整参数-adjust parameters')

threshold = st.sidebar.slider(
    '区别的深浅程度,darkness',
    0, 255, 50)
area = st.sidebar.slider(
    '区别的大小程度,area',
    0, 100, 10)
image_place4 = st.empty()
with image_place4:
    opencv_diff_img = cv2.subtract(open_cv_image1,open_cv_image2)
    diff_threshold = cv2.threshold(opencv_diff_img,threshold,255,cv2.THRESH_BINARY)[1]
    print(diff_threshold.shape)
    diff_threshold = cv2.cvtColor(diff_threshold, cv2.COLOR_BGR2GRAY)
    contours = cv2.findContours(diff_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(diff_threshold.shape, dtype='uint8')
    filled_after = mask.copy()

    for c in contours:
        area_con = cv2.contourArea(c)
        if area_con > area*10:
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(filled_after, [c], 0, (255,255,255), -1)

        st.image(toImgPIL(filled_after),use_column_width=True,caption="区别图,difference")

