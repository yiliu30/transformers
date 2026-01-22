import torch
import psutil
from threading import Lock
import logging
from loguru import logger


class MemoryUsageContext:
    """Context manager to record RAM and VRAM at entry and exit and log diffs.

    Usage:
        with MemoryUsageContext(label="quantize_block", device_list=["cuda:0", "cuda:1"]) as mm:
            ...
        # Logs start/end/current and diff via logger.
    """

    def __init__(self, label: str = "", device_list=None):
        self.label = label
        self.device_list = device_list
        self.start_ram = 0.0
        self.start_vram = {}
        self.end_ram = 0.0
        self.end_vram = {}

    def _normalize_devices(self):
        if self.device_list is None:
            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count() or 1))
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return list(range(torch.xpu.device_count() or 1))
            return []
        if not isinstance(self.device_list, (list, tuple)):
            return [self.device_list]
        return list(self.device_list)

    def _capture(self):
        process = psutil.Process()
        ram = process.memory_info().rss / 1024**3
        vram = {}
        devices = self._normalize_devices()
        if torch.cuda.is_available():
            indices = []
            for d in devices:
                s = str(d)
                if ":" in s:
                    try:
                        indices.append(int(s.split(":")[-1]))
                    except Exception:
                        continue
                elif s == "cuda":
                    indices.append(0)
                else:
                    try:
                        indices.append(int(d))
                    except Exception:
                        continue
            if not indices:
                indices = list(range(torch.cuda.device_count() or 1))
            for i in indices:
                vram[str(i)] = torch.cuda.memory_reserved(i) / 1024**3
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            indices = []
            for d in devices:
                s = str(d)
                if ":" in s:
                    try:
                        indices.append(int(s.split(":")[-1]))
                    except Exception:
                        continue
                elif s == "xpu":
                    indices.append(0)
                else:
                    try:
                        indices.append(int(d))
                    except Exception:
                        continue
            if not indices:
                indices = list(range(torch.xpu.device_count() or 1))
            for i in indices:
                vram[str(i)] = torch.xpu.memory_reserved(i) / 1024**3
        return ram, vram

    def __enter__(self):
        memory_monitor.update(device_list=self.device_list)
        self.start_ram, self.start_vram = self._capture()
        return self

    def __exit__(self, exc_type, exc, tb):
        memory_monitor.update(device_list=self.device_list)
        self.end_ram, self.end_vram = self._capture()
        ram_diff = round(self.end_ram - self.start_ram, 2)
        vram_diff_items = []
        all_keys = set(self.start_vram.keys()) | set(self.end_vram.keys())
        for k in sorted(all_keys):
            start = self.start_vram.get(k, 0.0)
            end = self.end_vram.get(k, 0.0)
            vram_diff_items.append(f"'{k}': {round(end - start, 2)}GB")

        start_vram_str = (
            ", ".join([f"'{k}': {round(v, 2)}GB" for k, v in sorted(self.start_vram.items())])
            if len(self.start_vram) > 0
            else ""
        )
        end_vram_str = (
            ", ".join([f"'{k}': {round(v, 2)}GB" for k, v in sorted(self.end_vram.items())])
            if len(self.end_vram) > 0
            else ""
        )
        diff_vram_str = ", ".join(vram_diff_items) if len(vram_diff_items) > 0 else ""

        msg = f"[mem-ctx:{self.label}] start_ram={round(self.start_ram, 2)}GB"
        if start_vram_str:
            msg += f", start_vram={{{{ {start_vram_str} }}}}"
        msg += f", end_ram={round(self.end_ram, 2)}GB"
        if end_vram_str:
            msg += f", end_vram={{{{ {end_vram_str} }}}}"
        msg += f", diff_ram={ram_diff}GB"
        if diff_vram_str:
            msg += f", diff_vram={{{{ {diff_vram_str} }}}}"

        memory_monitor.log_summary(msg)



class MemoryMonitor:
    """Global memory monitor for tracking peak RAM and VRAM usage."""

    _instance = None
    _lock = Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.peak_ram = 0.0  # GB
        self.peak_vram = {}  # {device_id: peak_mb}
        self.enabled = True

    def update(self, device_list=None):
        """Update current memory usage and track peaks."""
        if not self.enabled:
            return
        # Track RAM
        process = psutil.Process()
        current_ram = process.memory_info().rss / 1024**3  # GB
        self.peak_ram = max(self.peak_ram, current_ram)
        if device_list is None:  # TODO this has issue, wait for clean_memory all pass device_list
            device_list = [0]
        if device_list is not None:
            if not isinstance(device_list, (list, tuple)):
                device_list = [device_list]
        else:
            if torch.cuda.is_available():
                device_list = list(range(torch.cuda.device_count()))
            elif torch.xpu.is_available():
                device_list = list(range(torch.xpu.device_count()))

        for device in device_list:
            if str(device) == "cpu":
                continue
            if torch.cuda.is_available():
                current_vram = torch.cuda.memory_reserved(device) / 1024**3  # GB
                if device == "cuda":
                    device = "0"
            elif torch.xpu.is_available():
                current_vram = torch.xpu.memory_reserved(device) / 1024**3  # GB
                if device == "xpu":
                    device = "0"
            else:
                return

            device = str(device).split(":")[-1]
            if current_vram > 0:
                if device not in self.peak_vram:
                    self.peak_vram[device] = 0.0

                self.peak_vram[device] = max(self.peak_vram[device], current_vram)

    def update_cpu(self):
        if not self.enabled:
            return
        process = psutil.Process()
        current_ram = process.memory_info().rss / 1024**3  # GB
        self.peak_ram = max(self.peak_ram, current_ram)

    def reset(self):
        """Reset all statistics."""
        self.peak_ram = 0.0
        self.peak_vram = {}

    def get_summary(self):
        """Get summary of peak memory usage."""
        summary = f"'peak_ram': {round(self.peak_ram, 2)}GB"
        if len(self.peak_vram) > 0:
            sorted_items = sorted(self.peak_vram.items())
            if len(self.peak_vram) == 1:
                key, value = sorted_items[0]
                summary += f", 'peak_vram': {round(value, 2)}GB"
            else:
                items_str = ", ".join([f"'{k}': {round(v, 2)}GB" for k, v in sorted_items])
                summary += f", 'peak_vram': {{{items_str}}}"
        cur_ram = psutil.Process().memory_info().rss / 1024**3  # GB
        cur_vram = {}
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                cur_vram[str(device)] = round(torch.cuda.memory_reserved(device) / 1024**3, 2)
        elif torch.xpu.is_available():
            for device in range(torch.xpu.device_count()):
                cur_vram[str(device)] = round(torch.xpu.memory_reserved(device) / 1024**3, 2)
        summary += f", 'current_ram': {round(cur_ram, 2)}GB"
        if len(cur_vram) > 0:
            if len(cur_vram) == 1:
                key, value = list(cur_vram.items())[0]
                summary += f", 'current_vram': {value}GB"
            else:
                items_str = ", ".join([f"'{k}': {v}GB" for k, v in cur_vram.items()])
                summary += f", 'current_vram': {{{items_str}}}"
        return summary

    def log_summary(self, msg: str = ""):
        """Log memory usage summary."""
        summary = self.get_summary()
        logger.info(f"{msg} {summary}")
        return summary


# Global singleton instance
memory_monitor = MemoryMonitor()