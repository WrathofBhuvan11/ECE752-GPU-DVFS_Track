from m5.params import *
from m5.objects.SimObject import SimObject
from m5.objects.ClockDomain import SrcClockDomain

class GpuDVFSHandler(SimObject):
    type = 'GpuDVFSHandler'
    cxx_header = "sim/gpu_dvfs_handler.hh"
    cxx_class = "gem5::GpuDVFSHandler"

    domains = VectorParam.SrcClockDomain([], "List of domains to control")
    sys_clk_domain = Param.SrcClockDomain("System clock domain")
    enable = Param.Bool(False, "Enable/Disable the handler")
    
    # The transition latency depends on how much time the PLLs and voltage
    # regualators takes to migrate from current levels to the new level, is
    # usally variable and hardware implementation dependent. In order to
    # accomodate this effect with ease, we provide a fixed transition latency
    # associated with all migrations. Configure this to maximum latency that
    # the hardware will take to migratate between any two perforamnce levels.
    transition_latency = Param.Latency(
                "100us", "Latency for perf level transition"
    )
    
    # Generic SimObject pointer to the GPU
    shader = Param.SimObject(NULL, "Pointer to the GPU Shader object")

    def __init__(self, **kwargs):
        super(GpuDVFSHandler, self).__init__(**kwargs)
        print("PYTHON: GpuDVFSHandler instantiated in configuration!")
