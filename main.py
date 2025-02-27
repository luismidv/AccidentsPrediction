from fastapi import FastAPI
import joblib 
import numpy as np
import pandas as pd
import uvicorn
from pydantic import BaseModel
import dataprep as dp

app = FastAPI()

class Item(BaseModel):
    date : str
    countries : str
    local : str  
    accident_level : str 
    potential_accident_level : str
    genre : str 
    employee_ou_terceiro : str 
    risco_critico : str


@app.post("/predict/")
async def predict(item: Item):

    
    #CALL MODEL INFERENCE WITH NEW DATA
    new_data = pd.DataFrame([[
        item.date, item.countries, item.countries, item.accident_level, item.potential_accident_level, item.genre, item.employee_ou_terceiro, item.risco_critico
        ]], columns=['Data', 'Countries', 'Local',  'Accident Level',
        'Potential Accident Level', 'Genre', 'Employee ou Terceiro', 'Risco Critico'])
    
    new_data = new_data.reindex(columns = data_extractor.encoded_columns, fill_value = 0)

    try:
        prediction = data_extractor.new_predictions(new_data)
        decoded_prediction = data_extractor.get_encoding_info(prediction)
        print(decoded_prediction)
    except Exception as error:
        return {"prediction": error}


route = "./archive/IHMStefanini_industrial_safety_and_health_database.csv"
data_extractor = dp.DataExtractor()
data_extractor.datapreparer(route)
data_extractor.data_replacing()
models = data_extractor.model_charging()
data = data_extractor.grid_preprocessor(models)
prediction = data_extractor.model_selection(data, models)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port = 8000)
