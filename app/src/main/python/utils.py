import platform,sys,os
import numpy as np
import cv2
import torch
import PIL
from PIL import Image
from os.path import dirname, join
from com.chaquo.python import Python
import io
import base64

def get_info():
    ops=platform.system()
    rls=platform.release()
    pv=sys.version
    npv=np.__version__
    cvv=cv2.__version__
    tv=torch.__version__
    cd=os.getcwd()
    pilv=PIL.__version__
    dir=str(Python.getPlatform().getApplication().getFilesDir())

    return f'''
        cur dir: {dir}
        os: {ops}
        numpy: {npv}
        python: {pv[:6]}
        opencv: {cvv},
        torch: {tv}
        pillow: {pilv}

        '''

def exec_code(code):
    dir=str(Python.getPlatform().getApplication().getFilesDir())
    filename=join(dirname(dir),'file.txt')

    try:
        orginal_std=sys.stdout
        sys.stdout=open(filename,'w',encoding='utf8',errors='ignore')
        exec(code)
        sys.stdout.close()
        sys.stdout=orginal_std

        output=open(filename,'r').read()

    except Exception as e:
        sys.stdout=orginal_std
        output=e

    return str(output)




def sort(array):
    data=list(array)
    data=sorted(data,reverse=False)
    return data

def process_bitmap(bitmap_bytes):
    # Convert byte array to numpy array
    nparr = np.frombuffer(bitmap_bytes, np.uint8)


    # Decode numpy array to OpenCV image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform image processing (e.g., convert to grayscale)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert OpenCV image to PIL image
    pil_img = Image.fromarray(gray_img)

    # Convert PIL image back to byte array
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    imgs_str=base64.b64encode(byte_arr)

    return ""+str(imgs_str,'utf8')
