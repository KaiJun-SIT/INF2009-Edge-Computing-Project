# main.py
from modes.driving import DrivingMode
from modes.parking import ParkingMode
from utils.sensor_manager import SensorManager
from utils.resource_monitor import ResourceMonitor

class VehicleSystem:
    def __init__(self):
        self.sensor_manager = SensorManager()
        self.resource_monitor = ResourceMonitor()
        self.current_mode = None
        
    def switch_mode(self, mode_type):
        # Safely switch between driving and parking modes
        if self.current_mode:
            self.current_mode.stop()
            
        if mode_type == 'driving':
            self.current_mode = DrivingMode(self.sensor_manager)
        else:
            self.current_mode = ParkingMode(self.sensor_manager)
            
        self.current_mode.start()
        
    def run(self):
        try:
            # Start with driving mode
            self.switch_mode('driving')
            
            # Main event loop
            while True:
                # Monitor system resources
                if self.resource_monitor.is_overloaded():
                    self.current_mode.optimize_resources()
                    
                # Process mode switching triggers
                if self.detect_mode_change():
                    self.switch_mode(self.determine_new_mode())
                    
        except KeyboardInterrupt:
            if self.current_mode:
                self.current_mode.stop()

if __name__ == "__main__":
    system = VehicleSystem()
    system.run()