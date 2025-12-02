Only in ECE752-GPU-DVFS/src/sim: GpuDVFSHandler.py
Files ECE752-GPU-DVFS/src/sim/SConscript
Files ECE752-GPU-DVFS/src/sim/clock_domain.hh 
Files ECE752-GPU-DVFS/src/sim/gpu_dvfs_handler.cc 
Files ECE752-GPU-DVFS/src/sim/gpu_dvfs_handler.hh 
Files ECE752-GPU-DVFS/configs/example/gpufs/mi300.py
Files ECE752-GPU-DVFS/configs/example/gpufs/system/system.py

Add argument --enable-gpu-dvfs

ECE752-GPU-DVFS/build/VEGA_X86/gem5.opt ECE752-GPU-DVFS/configs/example/gpufs/mi300.py --app ECE752-GPU-DVFS/gem5-resources-copy/src/gpu/square/bin.default/square.default \
--kernel gem5-resources/src/x86-ubuntu-gpu-ml/vmlinux-gpu-ml --disk-image gem5-resources/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-gpu-ml --enable-gpu-dvfs

