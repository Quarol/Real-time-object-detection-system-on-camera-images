import time

class Timer:
    def get_current_time() -> float:
        return time.perf_counter()
    
    def get_duration_from_starting_point(start_point: float) -> float:
        return Timer.get_current_time() - start_point
    
    def get_duration(start_point: float, end_point: float) -> float:
        return end_point - start_point