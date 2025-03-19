# modes/driving.py
from threading import Thread, Event
from queue import Queue

class DrivingMode:
    def __init__(self, sensor_manager):
        self.sensor_manager = sensor_manager
        self.data_queue = Queue()
        self.stop_event = Event()
        self.active_sensors = []
        
    def start(self):
        # Activate needed sensors based on mode
        self.active_sensors = [
            self.sensor_manager.activate_sensor('mmwave_front'),
            self.sensor_manager.activate_sensor('camera'),
            self.sensor_manager.activate_sensor('ir_side')
        ]
        
        # Start processing threads
        self.threads = [
            Thread(target=self._process_mmwave),
            Thread(target=self._process_camera),
            Thread(target=self._monitor_blindspots)
        ]
        
        for thread in self.threads:
            thread.start()
            
    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()
        for sensor in self.active_sensors:
            sensor.deactivate()

    def _process_mmwave(self):
        while not self.stop_event.is_set():
            data = self.active_sensors[0].read()
            # Process mmWave data
            self.data_queue.put(('mmwave', data))