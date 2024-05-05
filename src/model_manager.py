from uuid import uuid1 as getuuid
from uuid import UUID
import torch, os

from src.AI import ChatAI
from src.train import Trainer
from threading import Thread

from src.watcher import OnMyWatch

class ModelManager:
    FOLDER_Path = "models/"
    EXTENSION = ".model"
    
    models: dict[UUID, ChatAI] = dict()
    trainers: dict[UUID, Trainer] = dict()
    default_model_id: UUID

    def __init__(self):
        self.watcher = OnMyWatch(self.WatcherHandler)
        modelpathdir = os.path.join(os.getcwd(), self.FOLDER_Path)

        for path in os.listdir(modelpathdir):
            if (path.endswith(self.EXTENSION)):
                full_path = os.path.join(modelpathdir, path)
                self.load_model(full_path)
                
        if len(self.models.keys()) <= 0:
            raise FileNotFoundError("models are empty")
        
        print(self.models.keys())
        
    def start_watcher(self):
        self.watcher.run()
        
    def stop_watcher(self):
        self.watcher.stop()
        
    def WatcherHandler(self, model_path: str):
        self.load_model(model_path)
        
    def keys(self) -> list[UUID]:
        return list(self.models.keys())
    
    def ismodelexist(self, bot_id: UUID) -> bool:
        return bot_id in self.models
    
    def get_model(self, bot_id: UUID) -> ChatAI:
        if self.ismodelexist(bot_id):
            return self.models[bot_id]
        return None
    
    def get_latest_model(self):
        return self.models[self.default_model_id]
    
    def load_model(self, model_path: str):
        torch_data = torch.load(model_path)
        basename = os.path.basename(model_path).replace(self.EXTENSION, "")
        _, bot_id_str = basename.split("-id-")
        bot_id = UUID(bot_id_str)
        model = ChatAI(
                torch_data["input_size"],
                torch_data["hidden_size"],
                torch_data["output_size"],
                torch_data["model_state"],
                torch_data["intents"]
            )
        self.models[bot_id] = model
        self.default_model_id = bot_id
        
    def delete_model(self, bot_id: UUID):
        del self.models[bot_id]
        del self.train_model[bot_id]
        
    def train_model(self, intent: list[dict[str, str | list[str]]]) -> tuple[UUID, Thread]:
        new_id = getuuid()
        self.trainers[new_id] = Trainer(intent)
        
        return new_id, Thread(target=self.thread_start_train, args=(self, new_id))
        
    def thread_start_train(self, _id: UUID):
        self.trainers[_id].start_train()
        self.models[_id] = self.trainers[_id].get_model()
    