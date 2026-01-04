class Asset:
    """
    Base class for strategic assets used in mission planning.
    """
    def __init__(self, name, speed_kmh, max_range_km, stealth_factor, fuel_efficiency=1.0):
        self.name = name
        self.speed_kmh = float(speed_kmh)
        self.max_range_km = float(max_range_km)
        self.stealth_factor = float(stealth_factor) # 0.0 (visible) to 1.0 (invisible)
        self.fuel_efficiency = float(fuel_efficiency) # km per unit of fuel

    def to_dict(self):
        return {
            "name": self.name,
            "speed_kmh": self.speed_kmh,
            "max_range_km": self.max_range_km,
            "stealth_factor": self.stealth_factor,
            "fuel_efficiency": self.fuel_efficiency
        }

class UAV(Asset):
    def __init__(self):
        super().__init__("MQ-9 Reaper (UAV)", speed_kmh=400, max_range_km=1850, stealth_factor=0.8, fuel_efficiency=5.0)

class GroundVehicle(Asset):
    def __init__(self):
        super().__init__("Tactical SUV (Ground)", speed_kmh=80, max_range_km=600, stealth_factor=0.2, fuel_efficiency=10.0)

class Helicopter(Asset):
    def __init__(self):
        super().__init__("MH-60 Black Hawk (Helo)", speed_kmh=280, max_range_km=500, stealth_factor=0.4, fuel_efficiency=2.0)

class DiplomaticCar(Asset):
    def __init__(self):
        super().__init__("Diplomatic Sedan", speed_kmh=100, max_range_km=700, stealth_factor=0.6, fuel_efficiency=12.0)

# Registry for easy UI access
ASSET_REGISTRY = {
    "Drone (UAV)": UAV(),
    "Ground Team (SUV)": GroundVehicle(),
    "Air Support (Helo)": Helicopter(),
    "Covert Ops (Sedan)": DiplomaticCar()
}
