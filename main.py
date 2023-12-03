import requests
from PIL import Image
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64



app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class TextInput(BaseModel):
    text: str

API_URL = "https://api-inference.huggingface.co/models/SamLowe/roberta-base-go_emotions"
headers = {"Authorization": "Bearer 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"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


API_URL_Z = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers_Z = {"Authorization": "Bearer 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"}
def query_Zephyr(payload):
	response = requests.post(API_URL_Z, headers=headers_Z, json=payload)
	return response.json()


API_URL_image = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
headers_image = {"Authorization": "Bearer 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"}
def query_image(payload):
    response = requests.post(API_URL_image, headers=headers_image, json=payload)
    return response.content




def get_mood_prediction(context):
    output = query({
        "inputs": context,
    })
    return output

@app.post("/process-text/")
async def process_text(input: TextInput):
    try:
        mood_prediction = get_mood_prediction(input.text)


        index, ANSWER = allusive_fy(input.text)
        Summarised_context = ANSWER[0]["generated_text"][index:]
        print(Summarised_context)
        image_bytes = query_image({
            "inputs": "Generate a anime style image of this: " + Summarised_context,
        })
        print(image_bytes)
        image = Image.open(io.BytesIO(image_bytes))
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr)

        
        return {
            "mood_prediction": mood_prediction,
            "badge_image": img_base64.decode('utf-8') 
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


def allusive_fy(CONTEXT):
    Z_Query_Final = f"""<|system|>
        You will generate a concise metaphorical depiction of the CONTEXT using imagery</s>
        <|user|>
        {CONTEXT}</s>
        <|assistant|>"""

    Zephyr_B_Beta_Final_Generated_Response = query_Zephyr({"inputs": Z_Query_Final,})

    return len(Z_Query_Final), Zephyr_B_Beta_Final_Generated_Response