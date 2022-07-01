import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.write('Brain Classification')
st.header('Klasifikasi Apakah ada tumor pada otak atau tidak')
st.write('Upload gambar otak')

model=tf.keras.models.load_model('model_best.hdf5')
model2=tf.keras.models.load_model('model_nadam.h5')

data_img=st.file_uploader(label='upload gambar mri atau xray atau ct scan otak dalam format jpg', type=['jpg'])
if data_img is not None:
    im=Image.open(data_img)
    st.image(im)
    new_size=(256,256)
    img = im.resize(new_size)
    x = np.array(img) # untuk ubah image kedalam array
    x = np.expand_dims(x, axis=0) #Memperluas bentuk array misal 1D jadi 2D, 0 berarti baris/horizontal [[1, 2]]
    images = np.vstack([x])
    classes = model.predict(images) #gunakan model yang terbaik untuk predict images
    if st.button('Predict'):
        if classes[0][0] == 1:
            st.write('Healthy')
        else:
            st.write('Brain Tumor')
    else:
        st.write(' ')

