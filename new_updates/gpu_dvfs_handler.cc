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
        // Schedule the first decision check 100ms (100 billion ticks) into the future.
        // This gives the OS time to boot without us polling uselessly.
        // 100ms = 100 * 1,000 * 1,000 * 1,000 ticks (assuming 1ps)
        schedule(decisionEvent, curTick() + 100000000000); 
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
    // Entry Point: gpuShader (The "System->Shader" part)
    if (!gpuShader) {
        return 0;
    }

    // Traversal: Iterate over cuList
    for (auto *cu : gpuShader->cuList) {
        // Traversal: Iterate over wfList (2D Vector: [SIMD][Slot])
        // wavefront-wfList is std::vector<std::deque<Wavefront*>>-> changing to deque
        for (const auto &simd_waves : cu->wfList) {
            // Traversal: Iterate over specific wavefronts in this SIMD
            for (auto *wave : simd_waves) {
                // Filter: only care about active waves (Not Stopped)
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
        schedule(decisionEvent, curTick() + 10000000000); // Retry in 10ms
        return;
    }

    // Get the current Program Counter
    Addr current_pc = readGpuPC();

    // ------------------------------------------------------------------
    // ADAPTIVE POLLING LOGIC
    // ------------------------------------------------------------------
    Tick nextPollTick;

    if (current_pc == 0) {
        // Case 1: GPU is IDLE (Booting or Waiting).
        // Do NOT check again for a long time (e.g., 10ms).
        // This fixes the "stuck" boot issue.
        // inform("GPU_DVFS: GPU Idle. Sleeping for 10ms...");
        nextPollTick = 10000000000; 
        
        // Schedule and exit immediately (no need to change perf level)
        schedule(decisionEvent, curTick() + nextPollTick);
        return;
    } else {
        // Case 2: GPU is ACTIVE (Running a Kernel).
        // We must poll fast to catch phase changes.
        // 10us = 10,000,000 ticks.
        nextPollTick = 10000000; 
        inform("GPU_DVFS: Captured Real Wavefront PC: %#x", current_pc);
    }
    // ------------------------------------------------------------------    

    // ------------------------------------------------------------------
    // Mod-15000 Cyclical Logic on GPU PC
    // ------------------------------------------------------------------
    // define a "block" of execution as 15000 PC increments.
    // Block 0 (PC 0-14999) -> Level 0 (High Perf)
    // Block 1 (PC 15000-29999) -> Level 1 (Med Perf) ...
    // ------------------------------------------------------------------
    
    uint64_t block_index = current_pc / 15000;
    PerfLevel desiredLevel = block_index % 3;

    PerfLevel currentLevel = domain->perfLevel();

    // Actuate: Only change settings if the desired level differs from current
    if (desiredLevel != currentLevel) {
        inform("GPU_DVFS: Switching Level %d -> %d at PC %#x", currentLevel, desiredLevel, current_pc);

        // Create a separate event to perform the update
        auto *e = new UpdateEvent();
        e->handler = this;
        e->domainIDToSet = targetDomain;
        e->perfLevelToSet = desiredLevel;
        
        // Execute the update immediately
        e->updatePerfLevel();
    }

    // Reschedule based on whether we are active or idle
    schedule(decisionEvent, curTick() + nextPollTick);
}

// ----------------------------------------------------------------------
// Actuation Event: Physically changes the clock/voltage
// ----------------------------------------------------------------------
void GpuDVFSHandler::UpdateEvent::updatePerfLevel()
{
    // dump() forces gem5 to write current stats to file.
    // reset() clears the counters.
    // This creates "buckets" of stats for each frequency phase.
    statistics::dump();
    statistics::reset();

    // Retrieve the domain and set the new level
    auto d = handler->findDomain(domainIDToSet);
    
    // This call modifies the SrcClockDomain's period and the VoltageDomain's voltage
    d->perfLevel(perfLevelToSet);

} 

} // namespace gem5

