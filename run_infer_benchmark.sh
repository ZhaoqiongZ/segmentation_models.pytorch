# ipexrun --throughput_mode --log_path ./logs unetpp_infer.py

ipexrun --ninstances 16  --ncore_per_instance 7 --log_path ./logs_16instances unetpp_infer.py