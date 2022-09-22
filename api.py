# FastAPI app to serve the model
from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from pydantic import BaseModel
from typing import List


class UserPrediction(BaseModel):
    user_id: str
    sequence_movie_ids: List[str]
    sequence_ratings: List[float]
    sex: str
    age_group: str
    occupation: str
    target_movie_id: str


app = FastAPI()

# Load v1 models
short_model = tf.keras.models.load_model("short_model/1")
long_model = tf.keras.models.load_model("long_model/1")


@app.post("/v1/short_model/predict")
def predict(user_data: UserPrediction):
    # Convert to appropriate model input format
    try:
        sequence_movie_ids = tf.convert_to_tensor(
            user_data.sequence_movie_ids, dtype=tf.string
        )
        sequence_ratings = tf.convert_to_tensor(
            user_data.sequence_ratings, dtype=tf.float32
        )
        input = {
            "user_id": tf.constant([user_data.user_id]),
            "sequence_movie_ids": tf.expand_dims(sequence_movie_ids, axis=0),
            "sequence_ratings": tf.expand_dims(sequence_ratings, axis=0),
            "sex": tf.constant([user_data.sex]),
            "age_group": tf.constant([user_data.age_group]),
            "occupation": tf.constant([user_data.occupation]),
            "target_movie_id": tf.constant([user_data.target_movie_id]),
        }
        prediction = short_model.predict(input).squeeze()[()].item()
        print(prediction)
        print(f"Type of prediction: {type(prediction)}")
        # print(f"Shape of prediction: {prediction.shape}")
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error occurred")


@app.post("/v1/long_model/predict")
def predict(user_data: UserPrediction):
    # Convert to appropriate model input format
    try:
        sequence_movie_ids = tf.convert_to_tensor(
            user_data.sequence_movie_ids, dtype=tf.string
        )
        sequence_ratings = tf.convert_to_tensor(
            user_data.sequence_ratings, dtype=tf.float32
        )
        input = {
            "user_id": tf.constant([user_data.user_id]),
            "sequence_movie_ids": tf.expand_dims(sequence_movie_ids, axis=0),
            "sequence_ratings": tf.expand_dims(sequence_ratings, axis=0),
            "sex": tf.constant([user_data.sex]),
            "age_group": tf.constant([user_data.age_group]),
            "occupation": tf.constant([user_data.occupation]),
            "target_movie_id": tf.constant([user_data.target_movie_id]),
        }
        prediction = long_model.predict(input).squeeze()[()].item()
        print(prediction)
        print(f"Type of prediction: {type(prediction)}")
        # print(f"Shape of prediction: {prediction.shape}")
        return {"prediction": prediction}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Error occurred")
