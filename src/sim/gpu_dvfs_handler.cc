#include "sim/gpu_dvfs_handler.hh"
#include "base/trace.hh"
#include "debug/GpuDVFS.hh"
#include "sim/stat_control.hh"

// definition of Shader for the dynamic_cast
#include "gpu-compute/shader.hh"
#include "gpu-compute/compute_unit.hh"

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
    // DEBUG PRINT 1: Constructor
    inform("GPU_DVFS: Handler Created. controlling %d domains.", domains.size());
}

// ----------------------------------------------------------------------
// Startup
// ----------------------------------------------------------------------
void GpuDVFSHandler::startup()
{
    // DEBUG PRINT 2: Startup
    inform("GPU_DVFS: Startup called.");

    if (enableHandler) {
        // Wait 100ms (100 billion ticks) for OS boot before first poll
        // This prevents DVFS from interfering with initialization
        inform("GPU_DVFS: Enabled! Scheduling first check in 0.5ms.");
        schedule(decisionEvent, curTick() + 5000000000); 
    } else {                                
        inform("GPU_DVFS: Handler is DISABLED in params. No logic will run.");
    }
}

// ----------------------------------------------------------------------
// Helper: Find a domain
// ----------------------------------------------------------------------
SrcClockDomain *
GpuDVFSHandler::findDomain(DomainID domain_id) const
{
    auto it = domains.find(domain_id);
    if (it == domains.end()) return nullptr;
    return it->second;
}

// ----------------------------------------------------------------------
// GLOBAL WAVEFRONT SCANNER (Data Collection Phase)
// ----------------------------------------------------------------------
std::map<Addr, int> GpuDVFSHandler::scanGlobalWavefrontState()
{
    std::map<Addr, int> pcHistogram;

    if (!gpuShader) {
        inform("GPU_DVFS ERROR: gpuShader pointer is NULL!");
        return pcHistogram;
    }

    int totalWavesSeen = 0;

    // Iterate over ALL Compute Units
    for (auto *cu : gpuShader->cuList) {
        // Iterate over ALL SIMDs
        for (const auto &simd_waves : cu->wfList) {
            // Iterate over ALL Wavefronts
            for (auto *wave : simd_waves) {
                totalWavesSeen++; 
                if (wave->getStatus() != Wavefront::S_STOPPED) {
                    // Add to histogram: This counts how many waves are at this specific PC
                    pcHistogram[wave->pc()]++;
                }
                ////else if (wave->getStatus() == Wavefront::S_WAIT_CNT) {
                ////    pcHistogram[wave->pc()].waitingCount++;
                ////}
            }
        }
    }

    //------------- will remove in the future- only temp debug----------------
    // DEBUG: Periodic Heartbeat for "Idle" states
    // Prints every 100,000 checks to prove it's scanning but finding nothing active
    static int idlePrintCounter = 0;
    if (pcHistogram.empty()) {
        idlePrintCounter++;
        if (idlePrintCounter % 100000 == 0) {
             inform("GPU_DVFS HEARTBEAT: Scanned %d total waves, but 0 are active. GPU is IDLE.", totalWavesSeen);
        }
    }
    //------------------------------------------------------------------------

    return pcHistogram;
}


// ----------------------------------------------------------------------
// THE PCSTALL GOVERNOR LOGIC (Decision Phase)
// ----------------------------------------------------------------------
void GpuDVFSHandler::runDecisionLoop()
{
    if (domains.empty()) {
        inform("GPU_DVFS ERROR: No domains registered to handler!");
        // Schedule check again later to avoid busy-loop crash, though this is fatal
        schedule(decisionEvent, curTick() + 10000000000); 
        return;
    }

    // Grab the first available domain (since we only have one GPU domain)
    auto it = domains.begin();
    SrcClockDomain *domain = it->second;
    DomainID targetDomain = it->first; 

    // DEBUG: Print what we found
    // inform("GPU_DVFS: Operating on Domain ID %d", targetDomain);

    if (!domain) {
        inform("GPU_DVFS ERROR: Domain pointer is null for ID %d", targetDomain);
        schedule(decisionEvent, curTick() + 10000000000);
        return;
    }

    // ------------------------------------------------------------------
    // 1. GATHER PHASE: Get the global distribution of PCs
    // ------------------------------------------------------------------
    std::map<Addr, int> pcMap = scanGlobalWavefrontState();

    // Poll period: 10us (10,000,000 ticks)
    Tick nextPollTick = 10000000;

    // Case 1: GPU is IDLE (Map is empty)
    if (pcMap.empty()) {
        // If GPU is idle, drop to min freq, or just sleep.
        schedule(decisionEvent, curTick() + 10000000000);
        return;
    }

    // ------------------------------------------------------------------
    // 2. ANALYZE PHASE: Calculate PC Concentration
    // ------------------------------------------------------------------
    int maxWavesAtOnePC = 0;
    int totalActiveWaves = 0;
    Addr dominantPC = 0;

    for (auto const& [pc, count] : pcMap) {
        totalActiveWaves += count;
        if (count > maxWavesAtOnePC) {
            maxWavesAtOnePC = count;
            dominantPC = pc;
        }
    }

    // "Concentration" metric: 0.0 to 1.0
    // High Concentration implies waves are synchronized at a bottleneck (Stall).
    // Low Concentration implies waves are executing freely (Compute).
    double concentration = 0.0;
    if (totalActiveWaves > 0) {
        concentration = (double)maxWavesAtOnePC / totalActiveWaves;
    }

    // DEBUG PRINT 4: Logic input
    inform("GPU_DVFS: Waves: %d | Concentration: %.2f | DomPC: %#x",
           totalActiveWaves, concentration, dominantPC);

    // ------------------------------------------------------------------
    // 3. DECISION PHASE: Predict Stall vs Compute
    // ------------------------------------------------------------------
    PerfLevel currentLevel = domain->perfLevel();
    PerfLevel desiredLevel = currentLevel;

    // > 50% Concentration -> STALL -> Low Freq (Level 2)
    // < 50% Concentration -> COMPUTE -> High Freq (Level 0)
    if (concentration > 0.5) {
        // PREDICTION: STALL
        // Strategy: Memory/Barrier bound. Lower Core Frequency to save energy.
        // Level 2 = Low Perf / Low Voltage
        desiredLevel = 2;
        DPRINTF(GpuDVFS, "PCStall: STALL DETECTED (C=%.2f at PC %#x). Target: Low Freq.\n",
                concentration, dominantPC);
    } else {
        // PREDICTION: COMPUTE / THROUGHPUT
        // Strategy: ALU bound. Raise Core Frequency to maximize throughput.
        // Level 0 = Max Perf / Max Voltage
        desiredLevel = 0;
        DPRINTF(GpuDVFS, "PCStall: COMPUTE DETECTED (C=%.2f). Target: Max Freq.\n",
                concentration);
    }

    // ------------------------------------------------------------------
    // 4. ACTUATION PHASE
    // ------------------------------------------------------------------
    if (desiredLevel != currentLevel) {
        inform("GPU_DVFS: Update %d -> %d | ActiveWaves: %d | Concentration: %.2f",
               currentLevel, desiredLevel, totalActiveWaves, concentration);

        auto *e = new UpdateEvent();
        e->handler = this;
        e->domainIDToSet = targetDomain; // Use the dynamically found ID
        e->perfLevelToSet = desiredLevel;
        e->updatePerfLevel();
    }

    // Schedule next check
    schedule(decisionEvent, curTick() + nextPollTick);
}


// ----------------------------------------------------------------------
// Actuation Event
// ----------------------------------------------------------------------
void GpuDVFSHandler::UpdateEvent::updatePerfLevel()
{
    // This dumping is what creates the multiple blocks in stats.txt
    statistics::dump();
    statistics::reset();

    auto d = handler->findDomain(domainIDToSet);
    d->perfLevel(perfLevelToSet);
} 

} // namespace gem5

