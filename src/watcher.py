from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Callable
import os, time
 
 
class OnMyWatch:
    watchDirectory = os.path.join(os.getcwd(), "models/")
 
    def __init__(self, model_callback: Callable[[str], None]):
        self.event_handler = Handler(model_callback)
        self.observer = Observer()
 
    def run(self):
        self.observer.schedule(self.event_handler, self.watchDirectory, recursive = True)
        self.observer.start()
 
    def stop(self):
        self.observer.stop()
        self.observer.join()
 
EXTENSION = ".model"

class Handler(FileSystemEventHandler):
    def __init__(self, model_callback: Callable[[str], None]) -> None:
        super().__init__()
        self.model_callback = model_callback
    
    def on_any_event(self, event):
        if event.is_directory:
            return None
 
        # elif event.event_type == 'created':
        #     # Event is created, you can process it now
        #     print("Watchdog received created event - % s." % event.src_path)
        elif event.event_type == 'modified':
            # Event is modified, you can process it now
            if (event.src_path.endswith(EXTENSION)):
                self.model_callback(event.src_path)
             
 
if __name__ == '__main__':
    def handler(file: str):
        print(file)
    watch = OnMyWatch(handler)
    watch.run()