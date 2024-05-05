from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.AI import ChatAI
from uuid import UUID

from src.model_manager import ModelManager

import uvicorn



app = FastAPI()
model_manager = ModelManager()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def getbotresponse(bot: ChatAI, prompt: str):
    response, status = bot.response(prompt)
    return {
        'botname': bot.bot_name,
        "response": response,
    }

class CreateModelReq(BaseModel):
    intents: list[dict[str, str | list[str]]]
    
class PromptReqModel(BaseModel):
    prompt: str


@app.get("/trainer/{bot_id}")
def status(bot_id: UUID):
    return {
        'id': bot_id,
        'message': model_manager.trainers[bot_id].status
    }
    
@app.post('/trainer')
async def create(Req: CreateModelReq, bg_task: BackgroundTasks):
    try:
        bot_id, train = model_manager.train_model(Req.intents)
    except ValueError as ve:
        return {
            "message": f'intent {ve}'
        }
    train.start()
    
    return {
        "id": bot_id,
        "message": "Created model. check status later"
    }

@app.delete('/trainer/{bot_id}')
def delete_model(bot_id: UUID):
    model_manager.delete_model(bot_id)
    return {
        'message': f'Delete okay - {bot_id}'
    }
    
@app.get("/models")
def getmodels():
    return {
        "models": model_manager.keys()
    }

    
@app.post("/prompt/{bot_id}")
def prompt(bot_id: UUID, Req: PromptReqModel):
    if model_manager.ismodelexist(bot):
        return {
            'id': {bot_id},
            'message': 'bot id not existing'
        }
        
    bot = model_manager.get_model(bot_id)
    return getbotresponse(bot, Req.prompt)
    
@app.post("/default-prompt/")
def l_prompt(Req: PromptReqModel):
    try:
        bot = model_manager.get_latest_model()
        return getbotresponse(bot, Req.prompt)
    except Exception as err:
        print(err)
        


"""_summary_
Api Routes:
GET /trainer/:id - get model status
POST /trainer - create model
DELETE /trainer/:id - delete model

GET /models - returns list of models

POST /prompt - get response prompt
POST /default-prompt - get response from default bot
"""

model_manager.start_watcher()