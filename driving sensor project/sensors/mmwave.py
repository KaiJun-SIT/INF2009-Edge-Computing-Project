# sensors/mmwave.py
import asyncio
import serial
import json
from typing import List
import numpy as np
from .import BaseSensor, SensorReading, ObjectDetection

class MMWaveRadar(BaseSensor):
    def __init__(self, sensor_id: str, config: dict):
        super().__init__(sensor_id, config)
        self.port = config['port']
        self.baud_rate = config['baud_rate']
        self.serial_connection = None
        self.read_buffer = bytearray()

    async def initialize(self) -> bool:
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1
            )
            # Send initialization commands to radar
            await self._configure_radar()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize MMWave radar: {e}")
            return False

    async def _configure_radar(self):
        # Configuration commands specific to your mmWave radar model
        config_commands = [
            "sensorStop",
            "flushCfg",
            "dfeDataOutputMode 1",
            "channelCfg 15 7 0",
            "adcCfg 2 1",
            "adcbufCfg -1 0 1 1 1",
            "profileCfg 0 60 359 7 57.14 0 0 70 1 256 5209 0 0 158",
            "chirpCfg 0 0 0 0 0 0 0 1",
            "frameCfg 0 0 16 0 100 1 0",
            "lowPower 0 0",
            "guiMonitor -1 1 1 1 0 0 1",
            "sensorStart"
        ]
        
        for cmd in config_commands:
            self.serial_connection.write(f"{cmd}\n".encode())
            await asyncio.sleep(0.1)

    async def read(self) -> List[SensorReading]:
        if not self.is_active:
            return []

        try:
            # Read raw data from radar
            data = await self._read_raw_data()
            if not data:
                return []

            # Process the raw data into detections
            detections = await self._process_radar_data(data)
            
            # Convert detections to sensor readings
            readings = []
            for det in detections:
                reading = SensorReading(
                    timestamp=asyncio.get_event_loop().time(),
                    sensor_id=self.sensor_id,
                    distance=det.distance,
                    confidence=det.confidence,
                    additional_data={
                        'angle': det.angle,
                        'velocity': det.velocity,
                        'object_type': det.object_type
                    }
                )
                readings.append(reading)
            
            return readings

        except Exception as e:
            self.logger.error(f"Error reading from MMWave radar: {e}")
            return []

    async def _read_raw_data(self) -> bytes:
        if self.serial_connection.in_waiting:
            return self.serial_connection.read(self.serial_connection.in_waiting)
        return b''

    async def _process_radar_data(self, raw_data: bytes) -> List[ObjectDetection]:
        # Process the raw radar data into object detections
        # This is a simplified example - actual implementation would depend on
        # your specific radar module's data format
        detections = []
        try:
            # Parse the raw data according to your radar's protocol
            # This is a placeholder for actual parsing logic
            detected_objects = self._parse_radar_protocol(raw_data)
            
            for obj in detected_objects:
                detection = ObjectDetection(
                    object_id=obj.get('id', 0),
                    object_type=obj.get('type', 'unknown'),
                    distance=obj.get('distance', 0.0),
                    angle=obj.get('angle', 0.0),
                    velocity=obj.get('velocity', 0.0),
                    confidence=obj.get('confidence', 0.8)
                )
                detections.append(detection)
                
        except Exception as e:
            self.logger.error(f"Error processing radar data: {e}")
            
        return detections

    def _parse_radar_protocol(self, raw_data: bytes) -> List[dict]:
        # Implement actual parsing logic for your radar's protocol
        # This is a placeholder
        return [{'id': 1, 'type': 'vehicle', 'distance': 10.0, 
                'angle': 45.0, 'velocity': 5.0, 'confidence': 0.9}]

    async def shutdown(self) -> None:
        if self.serial_connection:
            try:
                # Send stop command to radar
                self.serial_connection.write(b"sensorStop\n")
                await asyncio.sleep(0.1)
                self.serial_connection.close()
            except Exception as e:
                self.logger.error(f"Error shutting down MMWave radar: {e}")