"""Sample process RAM and device memory. Peaks cover this measured interval only."""

import sys
import threading
import time

GIB = 1024 ** 3


class ResourceMonitor:
    """RSS is host DRAM; CUDA allocated/reserved and driver used are distinct.

    Driver used includes all processes on the device. Sampled peaks may miss
    short spikes. PyTorch peaks include live weights, not just incremental use.
    """

    def __init__(self, interval=0.2):
        self.interval = interval
        self.record = {"sampling_interval_seconds": interval, "sample_count": 0,
                       "process_rss_peak_sampled_gib": None, "host_available_min_sampled_gib": None,
                       "gpu_driver_used_peak_sampled_gib": None}
        self.torch = sys.modules.get("torch")
        self.cuda = bool(self.torch is not None and self.torch.cuda.is_available())
        self.stop = threading.Event()
        try:
            import psutil
            self.process, self.psutil = psutil.Process(), psutil
            self.record["host_total_ram_gib"] = psutil.virtual_memory().total / GIB
        except ImportError:
            self.process = None
            self.record["host_ram_measurement"] = "unavailable: psutil missing"

    def _sample(self):
        self.record["sample_count"] += 1
        if self.process is not None:
            rss = self.process.memory_info().rss / GIB
            available = self.psutil.virtual_memory().available / GIB
            self.record.setdefault("process_rss_start_gib", rss)
            self.record["process_rss_end_gib"] = rss
            self.record["process_rss_peak_sampled_gib"] = max(
                rss, self.record["process_rss_peak_sampled_gib"] or 0)
            old = self.record["host_available_min_sampled_gib"]
            self.record["host_available_min_sampled_gib"] = min(available, old if old is not None else available)
        if self.cuda:
            free, total = self.torch.cuda.mem_get_info()
            used = (total - free) / GIB
            self.record["gpu_total_gib"] = total / GIB
            self.record["gpu_driver_used_peak_sampled_gib"] = max(
                used, self.record["gpu_driver_used_peak_sampled_gib"] or 0)

    def _poll(self):
        while not self.stop.wait(self.interval):
            try:
                self._sample()
            except Exception as error:
                self.record["sampling_error"] = type(error).__name__ + ": " + str(error)
                return

    def __enter__(self):
        self.started = time.perf_counter()
        if self.cuda:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
            self.record["gpu_allocated_start_gib"] = self.torch.cuda.memory_allocated() / GIB
            self.record["gpu_reserved_start_gib"] = self.torch.cuda.memory_reserved() / GIB
        self._sample()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join()
        try:
            self._sample()
            if self.cuda:
                self.record.update(gpu_allocated_peak_gib=self.torch.cuda.max_memory_allocated() / GIB,
                                   gpu_reserved_peak_gib=self.torch.cuda.max_memory_reserved() / GIB,
                                   gpu_allocated_end_gib=self.torch.cuda.memory_allocated() / GIB,
                                   gpu_reserved_end_gib=self.torch.cuda.memory_reserved() / GIB)
        except Exception as error:
            self.record["sampling_error"] = type(error).__name__ + ": " + str(error)
        self.record.update(elapsed_seconds=time.perf_counter() - self.started,
                           execution="failed" if kind is not None else "completed",
                           interpretation="Interval only; RSS is process host RAM. Driver used includes all GPU processes. "
                           "Sampled peaks are lower bounds; CUDA allocator peaks include live weights.")
        return False
