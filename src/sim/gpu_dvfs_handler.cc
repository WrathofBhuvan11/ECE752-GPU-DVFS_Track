#include "sim/gpu_dvfs_handler.hh"
#include "base/trace.hh"
#include "debug/GpuDVFS.hh"
#include "sim/stat_control.hh"

// definition of Shader for the dynamic_cast
#include "gpu-compute/shader.hh"
#include "gpu-compute/compute_unit.hh"
#include <cmath>
#include "sim/core.hh" // For sim_clock::Frequency

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
      gpuShader(dynamic_cast<Shader *>(p.shader)),
      nextPollTick(p.polling_interval), // Load from python- sampling/polling rate
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
        // Initialize PCSTALL Table
        inform("GPU_DVFS: Enabled! Initializing PCSTALL Predictor Table...");
        // Default to 1.0 (High Sensitivity / Compute Bound) so to start at Max Freq
        for(int i = 0; i < 128; i++) {
            sensitivityTable[i] = 1.0;
        }
        
        // Clear history structures
        wavefrontLastPC.clear();
        lastCuInstCount.clear();
        lastWfInstCount.clear();
        lastWfSchCycles.clear();
        lastWfSchStalls.clear();

        // Wait 5ms for OS boot before first poll
        inform("GPU_DVFS: Enabled! Scheduling first check in 5ms.");
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
// CHECK IF GPU IS RUNNING
// ----------------------------------------------------------------------
int GpuDVFSHandler::checkIfGPUIsRunning()
{
    //chaanged use of IPC stats to Check physical wavefront status directly.
    if (!gpuShader) return 0;
    
    for (auto *cu : gpuShader->cuList) {
        for (const auto &simd_waves : cu->wfList) {
            for (auto *wave : simd_waves) {
                // If any wave is NOT stopped, the GPU is running
                if (wave->getStatus() != Wavefront::S_STOPPED) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

// ----------------------------------------------------------------------
// STATS DUMP FUNCTION 
// ----------------------------------------------------------------------
int GpuDVFSHandler::dumpImportantStatsToConsole()
{
    static double prevInstTotal[40] = {0};
    static double prevNumCycles[40] = {0};
    
    // Iterate over ALL Compute Units
    int index = 0;
    for (auto *cu : gpuShader->cuList) {
        double ipc = cu->stats.ipc.total();
        
        if(!std::isnan(ipc) && ipc > 0){
            double instr = cu->stats.numInstrExecuted.total();
            double deltaInstr = instr - prevInstTotal[index];
            
            double numCycles = cu->stats.totalCycles.total();
            double deltaNumCycles = numCycles - prevNumCycles[index];
            
            if(deltaInstr < 0) deltaInstr = instr;
            if(deltaNumCycles < 0) deltaNumCycles = numCycles;
            
            double deltaIPC = deltaInstr / deltaNumCycles;
            
            double voltage = cu->voltage();
            double freq = (double)cu->frequency();
            
            // Avoid division by zero
            double performance = freq * deltaIPC;
            double edp = 0.0;
            double ed2p = 0.0;
            if (performance > 0) {
                edp = (voltage * voltage) / performance;
                ed2p = (voltage * voltage) / (performance * performance);
            }

            // RETRIEVE SENSITIVITY (Handle case where map is empty initially)
            double printSens = 0.0;
            if (currentCuSensitivity.find(cu) != currentCuSensitivity.end()) {
                printSens = currentCuSensitivity[cu];
            }
            
            // NOTE: Reporting average predicted sensitivity from table roughly here?
            // Since this function loops over CUs, can't show per-PC sensitivity easily.
            // Theo, I just leave the old "sensitivity[index]" or use 0.0 if that array is gone.
            // Assuming I removed the old array 'sensitivity', print 0.0 or a placeholder. 
            // So if you wanna edit this let me know #TODO

            inform("GPU_DVFS_STATS: CU: %d, clock: %d, Cycles: %d, IPC: %f, IPC_delta: %f, CPI: %f, CPI_delta: %f, Frequency: %d, Voltage: %f, EDP: %f, ED2P: %f, Sensitivity: %f"
               , index
               , curTick()
               , cu->stats.totalCycles.total()
               , cu->stats.ipc.total()
               , deltaIPC
               , (cu->stats.ipc.total() > 0) ? (1.0 / cu->stats.ipc.total()) : 0
               , (deltaIPC > 0) ? (1.0 / deltaIPC) : 0
               , freq 
               , voltage
               , edp 
               , ed2p
               , printSens//add sensivity here 
            );
            
            prevInstTotal[index] = instr;
            prevNumCycles[index] = numCycles;
        }
        index++;
    }
    return 0;
}


/// ------------------------------------------------------------------
/// IMPLEMENTATION Of Decision Policy for PCStall 
/// ------------------------------------------------------------------
void GpuDVFSHandler::runDecisionLoop()
{
    // ------------------------------------------------------------------
    // 1. STATUS CHECK and DEFINE FREQUENCY
    // ------------------------------------------------------------------
    static int idleHeartbeat = 0;
    int isRunning = checkIfGPUIsRunning();
    
    // assume a single DVFS domain for simplicity. 
    // If there are multiple, one would map CUs to Domains and repeat this logic per Domain.
    if (domains.empty()) {
        inform("GPU_DVFS ERROR: No domains registered in Handler!");
        return;
    }

    auto it = domains.begin();
    SrcClockDomain *domain = it->second;
    
    // Get Current Frequency (Normalized to MHz for readable math)
    // gem5 returns frequency in Hz (for example 1,000,000,000 for 1GHz)
    // Frequency (Hz) = sim_clock::Frequency / period
    double currentFreqHz = (double)sim_clock::Frequency / (double)domain->clockPeriod();
    
    // Convert to MHz for your sensitivity math
    double currentFreqMHz = currentFreqHz / 1000000.0;
    
    // Protect against startup edge cases (period=0)
    if (currentFreqMHz <= 0.0) currentFreqMHz = 1000.0; 

    if (!isRunning) {
        // GPU is idle. Slow down polling to save simulation overhead.
        if (++idleHeartbeat % 1000 == 0) {
           // inform("GPU_DVFS: GPU Idle... heartbeat %d", idleHeartbeat);
        }
        // Poll again later (10x slower) or standard interval
        schedule(decisionEvent, curTick() + (nextPollTick * 10));
        return;
    }
    
    // Reset heartbeat if it is running
    idleHeartbeat = 0;

    // ------------------------------------------------------------------
    // 2. PRE-LOOP SETUP
    // ------------------------------------------------------------------
    // Dump stats from the previous epoch before overwriting the metrics
    dumpImportantStatsToConsole();

    // Reset Aggregation Metrics for this new epoch
    currentCuSensitivity.clear();
    currentDomainSensitivity = 0.0;
    
    double totalPredictedSensitivity = 0.0;
    int activeWaves = 0;

    // ------------------------------------------------------------------
    // 3. MAIN LOOP: Iterate CUs -> Wavefronts
    // ------------------------------------------------------------------
    for (auto *cu : gpuShader->cuList) {
        double cuMeasuredSens = 0.0; // Sum of actual work done (for debug/stats)
        double cuPredictedSens = 0.0; // Sum of predicted future work

        // Two stage loop for CUs and WFs
        // Iterate over all SIMDs (Vector Units) in the CU
        for (const auto &simd_waves : cu->wfList) {
            // Iterate over all Wavefronts in the SIMD
            for (auto *wf : simd_waves) {
                
                // PHASE A: MEASURE & TRAIN (Calculate S_WF actual) 
                // 1. Get current cumulative instruction count
                // Note: .total() returns the standard gem5 stat value
                double currentInsts = wf->stats.numInstrExecuted.total();
                
                // 2. Retrieve previous count to find Delta
                double prevInsts = 0.0;
                if(lastWfInstCount.find(wf) != lastWfInstCount.end()) {
                    prevInsts = lastWfInstCount[wf];
                }
                
                // 3. Save current count for the NEXT loop
                lastWfInstCount[wf] = currentInsts;

                // 4. Calculate Delta Instructions
                double deltaInsts = currentInsts - prevInsts;
                if (deltaInsts < 0) deltaInsts = 0; // Handle resets/overflows
                
                // 5. Sensitivity using stall modeling with sync cycles
                // Get the total cycles the wavefront has been scheduled for since it began.
                double currentSchCycles = wf->stats.schCycles.total();
                double prevSchCycles = 0.0;
                if (lastWfSchCycles.find(wf) != lastWfSchCycles.end()) {
                    prevSchCycles = lastWfSchCycles[wf];
                }
                // Find total scheduled cycles in just THIS epoch.
                double deltaSchCycles = currentSchCycles - prevSchCycles;
                lastWfSchCycles[wf] = currentSchCycles;

                // Get the total cycles the wavefront has been stalled.
                double currentSchStalls = wf->stats.schStalls.total();
                double prevSchStalls = 0.0;
                if (lastWfSchStalls.find(wf) != lastWfSchStalls.end()) {
                    prevSchStalls = lastWfSchStalls[wf];
                }
                // Find total stall cycles in just THIS epoch.
                double deltaSchStalls = currentSchStalls - prevSchStalls;
                lastWfSchStalls[wf] = currentSchStalls;

                // Define Sensitivity (S) as the fraction of time the wavefront was NOT stalled.
                double S_wf_measured = 0.0;
                if (deltaSchCycles > 0) {
                    // Calculate the stall fraction (0.0=no stall, 1.0=always stalled).
                    double stall_fraction = deltaSchStalls / deltaSchCycles;
                    // Clamp to [0, 1] for reporting anomalies.
                    if (stall_fraction > 1.0) stall_fraction = 1.0;
                    if (stall_fraction < 0.0) stall_fraction = 0.0;
                    // Sensitivity is the "Compute-Bound" fraction of time (100% - Stall %).
                    S_wf_measured = 1.0 - stall_fraction;
                }
                else{
                    // If no cycles elapsed, the wavefront was inactive; sensitivity is zero.
                    S_wf_measured = 0.0;
                }

                // 6. Update Global Max (Auto-Tuning)
                if (S_wf_measured > globalMaxSensitivity) {
                    globalMaxSensitivity = S_wf_measured;
                }

                // Accumulate measured work for this CU
                cuMeasuredSens += S_wf_measured;
                //DPRINTF(GpuDVFS, "WF %p: measured S=%.2f\n", wf, S_wf_measured);

                // 6. Update the History Table
                // attribute this performance to the PC where the wave STARTED the epoch.
                if (wavefrontLastPC.count(wf)) {
                    Addr prevPC = wavefrontLastPC[wf];
                    int idx = getIndex(prevPC);
                    // Exponential Moving Average (Alpha = 0.3)
                    // New_Value = (Old_Value * (1 - Alpha)) + (Measured * Alpha)
                    sensitivityTable[idx] = (0.7 * sensitivityTable[idx]) + (0.3 * S_wf_measured);
                }

                // PHASE B: PREDICT (Lookup S_WF future) 
                // only predict for active waves. Stopped waves contribute 0 load.
                if (wf->getStatus() != Wavefront::S_STOPPED) {
                    Addr currentPC = wf->pc();
                    
                    // 1. Save this PC so it can update the table next time
                    wavefrontLastPC[wf] = currentPC;

                    // 2. Lookup Predicted Sensitivity
                    int idx = getIndex(currentPC);
                    double S_wf_predicted = sensitivityTable[idx];

                    // 3. Aggregate
                    cuPredictedSens += S_wf_predicted;
                    ++activeWaves;
                } else {
                    // If wave is stopped, remove it from history map to save memory/confusion
                    wavefrontLastPC.erase(wf);
                }
            }
            //  --- END WF Loop ---
        }
        // --- END CU Loopp ---
        
        // Save CU metrics for the stats dump function
        currentCuSensitivity[cu] = cuPredictedSens;
        
        // Aggregate into Domain total
        totalPredictedSensitivity += cuPredictedSens;
    }

    // Save Domain metrics for stats dump
    currentDomainSensitivity = totalPredictedSensitivity;

    // ------------------------------------------------------------------
    // 4. DECISION PHASE (Aggregated S_Domain)
    // ------------------------------------------------------------------
    DomainID targetDomain = it->first;
    PerfLevel currentLevel = domain->perfLevel();
    PerfLevel desiredLevel = currentLevel;
    
    //---------------- Autotuning-based decision making------------------
    // Decay the max slightly to adapt to phase changes (moving from Compute -> Memory phase)
    globalMaxSensitivity *= DECAY_FACTOR; 
    
    // Ensure it doesn't drop to zero (sanity floor)
    if (globalMaxSensitivity < 0.1) globalMaxSensitivity = 0.1;
    
    // Define Dynamic Thresholds
    // 33% (1/3) of Max -> Nominal
    // 66% (2/3) of Max -> Turbo
    double threshold_med  = globalMaxSensitivity * 0.33;
    double threshold_high = globalMaxSensitivity * 0.66;
    
    // Calculate Average Sensitivity of ACTIVE units
    double avgSensitivity = (activeWaves > 0) ? (totalPredictedSensitivity / activeWaves) : 0.0;
    
    if (avgSensitivity > threshold_high) {
        desiredLevel = 0; // Max Performance (Highest Freq, Highest Voltage)
    } else if (avgSensitivity > threshold_med) {
        desiredLevel = 1; // Medium Performance (if available)
    } else {
        desiredLevel = 2; // Low Performance (Lowest Freq, Save Power)
    }
    //-------------------------------------------------------------------

    // ------------------------------------------------------------------
    // 5. ACTUATION PHASE
    // ------------------------------------------------------------------
    if (desiredLevel != currentLevel) {
        
        DPRINTF(GpuDVFS, "Decision: S_Domain=%.2f | Level %d -> %d\n", 
                totalPredictedSensitivity, currentLevel, desiredLevel);

        // Schedule the update event to account for transition latency if needed
        // (or execute immediately if latency is modeled inside the clock domain)
        auto *e = new UpdateEvent();
        e->handler = this;
        e->domainIDToSet = targetDomain;
        e->perfLevelToSet = desiredLevel;
        e->updatePerfLevel(); 
    }

    // ------------------------------------------------------------------
    // 6. SCHEDULING
    // ------------------------------------------------------------------
    schedule(decisionEvent, curTick() + nextPollTick);
}

// ----------------------------------------------------------------------
// Actuation Event
// ----------------------------------------------------------------------
void GpuDVFSHandler::UpdateEvent::updatePerfLevel()
{
    // This dumping is what creates the multiple blocks in stats.txt
    // statistics::dump();
    // statistics::reset();
    
    auto d = handler->findDomain(domainIDToSet);
    if (d) d->perfLevel(perfLevelToSet);
}

} // namespace gem5

