import cv2
import pytesseract
import numpy as np
import math
import time
import os, shutil
import base64
import json 
import requests
import threading
from pytesseract import Output
from typing import Tuple, Union
from deskew import determine_skew

local = True
pathResults = './results/'
backoffice_url = 'https://jellyfish-app-kkaj7.ondigitalocean.app/api'
# backoffice_url = 'http://0f92-2a02-a456-3f72-1-e901-7568-c8d0-1404.eu.ngrok.io/backoffice/public/api'
request_headers = {"Content-Type":"application/x-www-form-urlencoded", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,/;q=0.8,application/signed-exchange;v=b3;q=0.9"}

for tempImgName in os.listdir(pathResults):
    tempImagePath = os.path.join(pathResults, tempImgName)
    try:
        if os.path.isfile(tempImagePath) or os.path.islink(tempImagePath):
            os.unlink(tempImagePath)
        elif os.path.isdir(tempImagePath):
            shutil.rmtree(tempImagePath)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (tempImagePath, e))


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
 
def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def processTessract(text):
    data = {}
    print(text)
    wordNumCounter = 0
    lineCounter = 0
    for i in range(len(text['line_num'])):
        txt = text['text'][i]
        block_num = text['block_num'][i]
        word_num = text['word_num'][i]
        conf = text['conf'][i]
        if not (txt == '' or txt.isspace()):
            block = (txt, conf, block_num)
            if block_num in data:
                if word_num > wordNumCounter:
                    wordNumCounter += 1
                    data[block_num][lineCounter].append(block)
                else:
                    wordNumCounter = 0
                    lineCounter += 1
                    data[block_num][lineCounter] = [block]
            else:
                wordNumCounter = 0
                lineCounter += 1
                data[block_num] = {}
                data[block_num][lineCounter] = [block]

    idx = 0
    linedata = {}
    for _, b  in data.items():
        for _, l in b.items():
            linedata[idx] = l
            idx += 1

    line_idx = 0
    exportData = {}
    for _, line in linedata.items():
        exportData[line_idx] = line
        line_idx += 1

    return exportData
     
def processFrame(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    framq = rotate(frame, angle, (0, 0, 0))

    testFrame = frame
    imgFloat = testFrame.astype(np.float) / 255
    kChannel = 1 - np.max(imgFloat, axis=2)
    kChannel = (255 * kChannel).astype(np.uint8)

    binaryThresh = 75
    _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255 ,cv2.THRESH_BINARY)
    kernelSize = 1
    opIterations = 5
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
    binaryImageInverted = cv2.bitwise_not(binaryImage)
    
    if(local):
        imgPath = pathResults + str(time.time()) + '-frame.png'
        cv2.imwrite(imgPath, frame)
        # cv2.imwrite(pathResults + str(time.time()) + '-text-inverted.png', binaryImageInverted)

 
    d = pytesseract.image_to_data(frame, output_type=Output.DICT, config='--psm 4')
    non_empty_text = list(filter(lambda item: item != '', d['text']))
    # print(non_empty_text)

    d2= pytesseract.image_to_data(binaryImageInverted, output_type=Output.DICT, config='--psm 4')
    # print(d2)
    processedData = processTessract(d2)
    # print('inbin')
    # print(non_empty_text2)
    print(processedData)


    with open(imgPath, "rb") as f:
        im_bytes = f.read()        
    im_b64 = base64.b64encode(im_bytes).decode("utf8")
    items = {'data': json.dumps(processedData), 'receipt_image': im_b64}

    x = requests.post(backoffice_url + '/scans', headers=request_headers, data = items)
    print(x.text)

    # n_boxes = lend(['text'])
    # for i in range(n_boxes):
    #     if int(d['conf'][i]) > 60:
            # (text, x, y, w, h) = (d['text'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # don't show empty text
            # if text and text.strip() != "":
                # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # frame = cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

counter = 1
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # process_thread = threading.Thread(target=processFrame, name="FrameProcessor", args=[frame])
        # print('starting process')
        # process_thread.start()
        processFrame(frame)
 
    counter += 1
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()