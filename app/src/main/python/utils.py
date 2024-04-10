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
import ultralytics
from ultralytics import YOLO
import time

# /data/user/0/dev.jewel.machinelearning/files
DIR_FILES=str(Python.getPlatform().getApplication().getFilesDir()) 

def get_info():
    ops=platform.system()
    pv=sys.version
    npv=np.__version__
    cvv=cv2.__version__
    tv=torch.__version__
    
    pilv=PIL.__version__
    ultv=ultralytics.__version__


    return f'''
        cur dir: {DIR_FILES}
        os: {ops}
        numpy: {npv}
        python: {pv[:6]}
        opencv: {cvv},
        torch: {tv}
        pillow: {pilv}
        ultralytics: {ultv}

        '''

def exec_code(code):

    filename=join(dirname(DIR_FILES),'file.txt') 

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

def read_file():
    # src/main/python/test.txt
    # /data/data/dev.jewel.machinelearning/files/chaquopy/AssetFinder/app/test.txt
    filename=join(dirname(__file__),'test.txt')
    with open(filename,'r',encoding='utf8',errors='ignore') as f:
        data=f.read()
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

    return pil_to_base64(pil_img)

def pil_to_base64(img):
    # Convert PIL image back to byte array
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    imgs_str=base64.b64encode(byte_arr)
    return str(imgs_str,'utf8')

def image_info(img_path):
    img=Image.open(img_path)
    nparray=np.array(img)
    info=f"(H,W,C): {nparray.shape}"

    return info

def remove_bg(img_path):
    st=time.time()
    img=Image.open(img_path)
    filename=join(dirname(__file__),'yolov8s-seg.pt')
    model=YOLO(filename)
    results=model(img)
    mask=results[0].masks.data[0] # get first mask data[0]
    mask=(mask.numpy()*255).astype('uint8') # np to image format
    mask=Image.fromarray(mask).resize(img.size)
    new_img=Image.new("RGBA",img.size,0)
    new_img.paste(img,mask=mask)
    filename=join(DIR_FILES,'nobg.png')
    new_img.save(filename)
    print(f'process took: {int((time.time()-st)*1000)}ms')
    return filename


def gray_image(img_path):
    img=Image.open(img_path).convert("L")

    filename=join(DIR_FILES,'gray.jpg')
    img.save(filename)
    return filename
    