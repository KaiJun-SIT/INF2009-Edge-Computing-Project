# sensors/__init__.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

@dataclass
class SensorReading:
    """Base class for sensor readings"""
    timestamp: float
    sensor_id: str
    distance: float
    confidence: float = 1.0
    additional_data: Optional[Dict[str, Any]] = None

@dataclass
class ObjectDetection:
    """Class for detected objects"""
    object_id: int
    object_type: str
    distance: float
    angle: float
    velocity: Optional[float] = None
    confidence: float = 1.0

class BaseSensor(ABC):
    """Abstract base class for all sensors"""
    def __init__(self, sensor_id: str, config: dict):
        self.sensor_id = sensor_id
        self.config = config
        self.logger = logging.getLogger(f"sensor.{sensor_id}")
        self.is_active = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the sensor hardware"""
        pass

    @abstractmethod
    async def read(self) -> List[SensorReading]:
        """Read data from the sensor"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Safely shut down the sensor"""
        pass

    async def activate(self) -> bool:
        """Activate the sensor"""
        try:
            self.is_active = await self.initialize()
            return self.is_active
        except Exception as e:
            self.logger.error(f"Failed to activate sensor {self.sensor_id}: {e}")
            return False

    async def deactivate(self) -> None:
        """Deactivate the sensor"""
        if self.is_active:
            await self.shutdown()
            self.is_active = False