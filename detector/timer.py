import time

class Timer:
    def get_current_time():
        return time.perf_counter()
    
    def get_duration(start_point: float):
        return Timer.get_current_time() - start_point()