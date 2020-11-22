
#from perTrain.preTrain_evaluting import preTrain_E

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import cv2
from evalutor import * 

import base64
import numpy as np

from pydantic import BaseModel

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

def base64_to_cv2(image_base64):
    """base64 image to cv2"""
    image_bytes = base64.b64decode(image_base64)
    np_array = np.fromstring(image_bytes, np.uint8)
    image_cv2 = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image_cv2


def cv2_to_base64(image_cv2):
    """cv2 image to base64"""
    image_bytes = cv2.imencode('.jpg', image_cv2)[1].tostring()
    image_base64 = base64.b64encode(image_bytes).decode()
    return image_base64
    
class Response(BaseModel):
    testCode: str
    base64_image: str
    
@app.post("/recognition")
async def recognition(request:Response):

    #getting image
    input = base64_to_cv2(request.base64_image)
    input_path = 'Plate_examples/input.jpg'
    cv2.imwrite(input_path, input)
    
    #evaluting
    final_string = evaluting(wpod_net, model, labels, test_image_path=input_path)

    #img/plot_image.jpg
    #img/test_roi.jpg
    #img/crop_characters.jpg
    #img/character.jpg
    #img/final_result.jpg
    output_1 = cv2.imread('img/test_roi.jpg')
    output_2 = cv2.imread('img/final_result.jpg')
    
    output_1 = cv2_to_base64(output_1)
    output_2 = cv2_to_base64(output_2)
    
    return {"testcode":request.testCode,
            "output_1":output_1,
            "output_2":output_2,
            "result"  :final_string
            }
    
if __name__ == "__main__":
    print("start main server")
    #loading the model
    wpod_net, model, labels = initialize()
    
    #start the server
    uvicorn.run(app, host="0.0.0.0", port=5000)
    
    
    