import uvicorn
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import json
from PIL import Image
import io


app = FastAPI()

MODEL_PATH = "C://Project//models//efficientnet_finetuned.keras"
model = tf.keras.models.load_model(MODEL_PATH)

with open("C://Project//app//styles_info.json", "r", encoding="utf-8") as f:
    styles_info = json.load(f)



with open("C://Project//models//class_names.json", "r") as f:
    class_names = json.load(f)

print("Class sayısı:", len(class_names))


def prepare_image(file):
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = img.resize((224, 224))

    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)


    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = prepare_image(image_bytes)

        preds = model.predict(img)
        idx = int(np.argmax(preds[0]))
        
        predicted_class = class_names[idx]

        style_details = styles_info.get(predicted_class, {
        "description": "Bu stil için açıklama bulunamadı.",
        "pioneer": "Bilinmiyor"
})

        return JSONResponse({
            "predicted_index": idx,
            "predicted_class": class_names[idx],
            "confidence": float(np.max(preds[0])),
            "description": style_details["description"],
            "pioneer": style_details["pioneer"],
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


