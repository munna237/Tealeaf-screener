.PHONY: verify bench
verify:            ## check checksums and regenerate all image-free numbers (CPU, ~1 min)
	python tools/verify.py
bench:             ## latency + memory of the deployed model on this machine
	python deploy/benchmark_onnx.py --model models/tea_screener.onnx --threads 1
	python deploy/benchmark_onnx.py --model models/tea_screener.onnx --threads 0
