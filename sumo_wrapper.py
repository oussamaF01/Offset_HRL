import os
import subprocess
import time
import socket
import shutil
from itertools import count


_SUMO_CONNECTION_COUNTER = count()


class SumoMobilityWrapper:
    def __init__(
        self,
        config_path="scenario/mobility/sim.sumocfg",
        sumo_binary="sumo",
        port=8813,
        label=None,
        begin_time=0,
    ):
        self.config_path = config_path
        self.sumo_binary = sumo_binary
        self.port = port
        self.label = label or f"sumo_{next(_SUMO_CONNECTION_COUNTER)}"
        self.begin_time = begin_time
        self.started = False
        self.sumo_process = None
        self.traci = None

    def _find_free_port(self):
        """Find a free port to use for TraCI communication."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def _resolve_sumo_binary(self):
        binary = shutil.which(self.sumo_binary)
        if binary is not None:
            return binary

        if os.path.isabs(self.sumo_binary) and os.path.exists(self.sumo_binary):
            return self.sumo_binary

        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            candidate = os.path.join(sumo_home, "bin", self.sumo_binary)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            f"SUMO executable not found: {self.sumo_binary!r}. "
            "Install SUMO and make sure 'sumo' is on PATH, set SUMO_HOME, "
            "or pass an absolute sumo_binary path."
        )

    def start(self):
        if self.started:
            return

        try:
            import traci
        except ImportError as exc:
            raise ImportError(
                "SUMO TraCI support requires the 'traci' Python package. "
                "Install SUMO/TraCI before enabling use_sumo_mobility."
            ) from exc

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"SUMO config not found: {self.config_path}")

        sumo_binary = self._resolve_sumo_binary()

        # Find a free port for TraCI
        self.port = self._find_free_port()

        # Start SUMO with TraCI enabled
        try:
            self.sumo_process = subprocess.Popen(
                [
                    sumo_binary,
                    "-c",
                    self.config_path,
                    "--begin",
                    str(self.begin_time),
                    "--remote-port",
                    str(self.port),
                    "--ignore-route-errors",
                    "true",
                    "--no-step-log",
                    "true",
                    "--no-warnings",
                    "true",
                    "--duration-log.disable",
                    "true",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )

            # Connect traci to the running SUMO instance. A unique label lets
            # multiple city scenarios stay alive in one Python process.
            last_error = None
            for _ in range(40):
                try:
                    traci.init(port=self.port, label=self.label)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if self.sumo_process.poll() is not None:
                        break
                    time.sleep(0.05)
            if last_error is not None:
                raise last_error
            traci.switch(self.label)
            self.traci = traci
            self.started = True
        except Exception as e:
            if self.sumo_process:
                try:
                    self.sumo_process.terminate()
                except:
                    pass
            raise RuntimeError(f"Failed to start SUMO: {e}")

    def step(self):
        if not self.started:
            raise RuntimeError("SUMO has not been started. Call start() first.")

        self.traci.switch(self.label)
        self.traci.simulationStep()
        return self.get_mobility_state()

    def get_mobility_state(self):
        vehicles = {}
        persons = {}

        traci = self.traci
        traci.switch(self.label)

        for vehicle_id in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vehicle_id)
            vehicles[vehicle_id] = {
                "x": x,
                "y": y,
                "speed": traci.vehicle.getSpeed(vehicle_id),
                "angle": traci.vehicle.getAngle(vehicle_id),
                "road_id": traci.vehicle.getRoadID(vehicle_id),
            }

        get_person_angle = getattr(traci.person, "getAngle", None)
        get_person_road_id = getattr(traci.person, "getRoadID", None)

        for person_id in traci.person.getIDList():
            x, y = traci.person.getPosition(person_id)

            person_angle = None
            if callable(get_person_angle):
                try:
                    person_angle = get_person_angle(person_id)
                except Exception:
                    person_angle = None

            person_road_id = None
            if callable(get_person_road_id):
                try:
                    person_road_id = get_person_road_id(person_id)
                except Exception:
                    person_road_id = None

            persons[person_id] = {
                "x": x,
                "y": y,
                "speed": traci.person.getSpeed(person_id),
                "angle": person_angle,
                "road_id": person_road_id,
            }

        return {
            "vehicles": vehicles,
            "persons": persons,
        }

    def close(self):
        if self.started:
            try:
                self.traci.switch(self.label)
                self.traci.close()
            except Exception:
                pass
            self.started = False
            self.traci = None

        if self.sumo_process:
            try:
                self.sumo_process.terminate()
                self.sumo_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.sumo_process.kill()
            except Exception:
                pass
            self.sumo_process = None
