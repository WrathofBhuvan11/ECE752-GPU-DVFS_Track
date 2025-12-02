#include "sim/gpu_dvfs_handler.hh"
#include "base/trace.hh"
#include "debug/DVFS.hh"
#include "sim/stat_control.hh"

// definition of Shader for the dynamic_cast
#include "gpu-compute/shader.hh"

namespace gem5
{

// ----------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------
GpuDVFSHandler::GpuDVFSHandler(const Params &p)
   : SimObject(p),
   sysClkDomain(p.sys_clk_domain),
   enableHandler(p.enable),
   _transLatency(p.transition_latency),
   // Cast the generic SimObject pointer from Python to a Shader pointer
   gpuShader(dynamic_cast<Shader*>(p.shader)),
   // Initialize the decisionEvent to call 'runDecisionLoop' when triggered
   decisionEvent([this]{ runDecisionLoop(); }, name())
{
    // Populate the map of domains provided in the Python config
    for (auto *d : p.domains) {
        domains[d->domainID()] = d;
    }
}

// ----------------------------------------------------------------------
// Startup: Kicks off the autonomous loop
// ----------------------------------------------------------------------
void GpuDVFSHandler::startup()
{
    if (enableHandler) {
        // Schedule the first decision check 10000 ticks into the future.
        // This ensures the system is fully initialized before start looking at PCs.
        schedule(decisionEvent, curTick() + 10000); //#TODO Fix the right value
    }
}

// ----------------------------------------------------------------------
// Helper: Find a domain by ID
// ----------------------------------------------------------------------
SrcClockDomain *
GpuDVFSHandler::findDomain(DomainID domain_id) const
{
    auto it = domains.find(domain_id);
    if (it == domains.end())
        return nullptr;
    return it->second;
}

// ----------------------------------------------------------------------
// Gpu CU SPYING IMPLEMENTATION
// ----------------------------------------------------------------------
// GPU access : system-->gpuShader->cuList[0]->wfList[0]->pc();
Addr GpuDVFSHandler::readGpuPC()
{
    //  Entry Point: gpuShader (The "System->Shader" part)
    if (!gpuShader) {
        return 0;
    }
    //  Traversal: Iterate over cuList
    for (auto *cu : gpuShader->cuList) {
        // Traversal: Iterate over wfList (2D Vector: [SIMD][Slot])
        // wavefront-wfList is std::vector<std::vector<Wavefront*>>
        for (const auto &simd_waves : cu->wfList) {
            //  Traversal: Iterate over specific wavefronts in this SIMD
            for (auto *wave : simd_waves) {
                //  Filter: only care about active waves (Not Stopped)
                if (wave->getStatus() != Wavefront::S_STOPPED) {
                    // 6. Extraction: Grab the PC
                    return wave->pc();
                }
            }
        }
    }
    // If no active wavefronts are found (GPU is idle), return 0
    return 0;
}


// ----------------------------------------------------------------------
// THE GOVERNOR LOGIC LOOP
// ----------------------------------------------------------------------
void GpuDVFSHandler::runDecisionLoop()
{
    // Target the specific Domain ID for the GPU (Assuming ID 1 set in Python)
    DomainID targetDomain = 1; 
    auto *domain = findDomain(targetDomain);
    
    // Safety check: If domain isn't ready, wait and try again later
    if (!domain) {
        schedule(decisionEvent, curTick() + 10000);
        return;
    }

    //  Get the current Program Counter
    Addr current_pc = readGpuPC();

    // Print to terminal/log for verification
    if (current_pc > 0) {
        inform("GPU_DVFS: Captured Real Wavefront PC: %#x", current_pc);
    }
    //else {
    //    inform("GPU_DVFS: Captured IDLE PC....");
    //}
    

    // ------------------------------------------------------------------
    // Mod-150 Cyclical Logic on GPU PC; Level Cyclical Logic Policy
    // #TODO- We will swap this policy with PCSTALL subroutine
    // ------------------------------------------------------------------
    // define a "block" of execution as 150 PC increments.
    // cycle through 3 performance levels based on which block in.
    // Block 0 (PC 0-149)   -> Level 0 (High Perf)
    // Block 1 (PC 150-299) -> Level 1 (Med Perf)
    // Block 2 (PC 300-449) -> Level 2 (Low Perf)
    // Block 3 (PC 450-599) -> Level 0 ... repeats
    // ------------------------------------------------------------------
    
    uint64_t block_index = current_pc / 150;
    PerfLevel desiredLevel = block_index % 3;

    PerfLevel currentLevel = domain->perfLevel();

    //  Actuate: Only change settings if the desired level differs from current
    if (desiredLevel != currentLevel) {
        inform("GPU_DVFS: Switching Level %d -> %d", currentLevel, desiredLevel);
               
        // Create a separate event to perform the update
        // This decouples the decision logic from the hardware state change
        auto *e = new UpdateEvent();
        e->handler = this;
        e->domainIDToSet = targetDomain;
        e->perfLevelToSet = desiredLevel;
        
        // Execute the update immediately (or schedule it with latency if desired)
        e->updatePerfLevel(); 
    }

    // Reschedule: Run this logic loop again in 1000 ticks
    schedule(decisionEvent, curTick() + 1000); 
}

// ----------------------------------------------------------------------
// Actuation Event: Physically changes the clock/voltage
// ----------------------------------------------------------------------
void GpuDVFSHandler::UpdateEvent::updatePerfLevel()
{
    // dump() forces gem5 to write current stats to file. 
    // reset() clears the counters.
    // This creates "buckets" of stats for each frequency phase, allowing us
    // to verify that performance actually changed during that phase.
    statistics::dump();
    statistics::reset();
    
    // Retrieve the domain and set the new level
    auto d = handler->findDomain(domainIDToSet);
    
    // This call modifies the SrcClockDomain's period and the VoltageDomain's voltage
    d->perfLevel(perfLevelToSet);
}

} // namespace gem5
