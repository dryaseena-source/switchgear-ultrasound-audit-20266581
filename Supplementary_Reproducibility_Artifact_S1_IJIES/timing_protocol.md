# CPU timing protocol

Latency measurements are single-record forward-pass measurements in `torch.inference_mode()` on the stated CPU environment. Each trained checkpoint receives 100 warm-up forward passes followed by 300 timed forward passes with batch size 1. PyTorch is restricted to five CPU threads. Data loading, preprocessing, Python file I/O, and model construction are excluded. Median and 95th-percentile latency are reported; training wall time is reported separately. These CPU measurements are implementation-specific and are not interpreted as edge-device energy consumption.
