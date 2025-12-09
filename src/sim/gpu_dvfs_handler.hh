#ifndef __SIM_GPU_DVFS_HANDLER_HH__
#define __SIM_GPU_DVFS_HANDLER_HH__

#include "params/GpuDVFSHandler.hh"
#include "sim/sim_object.hh"
#include "sim/clock_domain.hh"
#include "sim/eventq.hh"
#include <map>
#include <vector>

// GPU Headers
#include "gpu-compute/shader.hh"
#include "gpu-compute/compute_unit.hh"
#include "gpu-compute/wavefront.hh"

namespace gem5
{

/**
 * GpuDVFSHandler
 * A specialized handler for managing GPU Dynamic Voltage and Frequency Scaling (DVFS).
 *
 * IMPLEMENTATION STRATEGY: PCSTALL (Predict, Don't React)
 * Reference: Bharadwaj et al
 * PCSTALL is "predictive". It assumes that the "Frequency Sensitivity" (compute-bound vs memory-bound)
 * of a wavefront is tied to the code it is executing. By tracking the Program Counter (PC),
 * one can figure which parts of the kernel need high frequency and which are stalled on memory.
 * -------------------------------------------------------
 */
class GpuDVFSHandler : public SimObject
{
  public:
    typedef GpuDVFSHandlerParams Params;
    GpuDVFSHandler(const Params &p);

    // Standard gem5 typedefs
    typedef SrcClockDomain::DomainID DomainID;
    typedef SrcClockDomain::PerfLevel PerfLevel;

    /**
     * startup()
     * Called by gem5 after all objects are created but before simulation starts.
     * Use this to schedule the first iteration of our decision loop.
     */
    void startup() override;

  private:
    // --- PCSTALL HARDWARE STRUCTURES ---
    //---------------------------------------------------------------------------------------------

    // 1. Prediction Table (Hardware Storage)
    // Concept: A small direct-mapped cache that remembers the "Sensitivity" of code blocks.
    // - High Sensitivity : Compute-Bound. The code scales with Frequency. (Predict: High Freq)
    // - Medium Sensitivity : will decide threshold
    // - Low Sensitivity : Memory-Bound. The code is waiting on RAM. (Predict: Low Freq)
    // Size: 128 entries is sufficient as GPU kernels loop over small code segments.
    double sensitivityTable[128];

    // 2. Indexing Logic
    // Ignore the lowest 4 bits (byte offsets) because instructions execute in blocks.
    // Mask to 127 to map the PC into our 128-entry table.
    int getIndex(Addr pc) const { return (pc >> 4) & 127; }

    // 3. History Tracking (For Learning)
    // To train the table,to know: "Where was this wavefront 1us ago?"
    // During the "Update Phase", attribute the *observed* performance of the last epoch
    // back to the PC stored here. This allows the predictor to "learn" from the past.
    std::map<Wavefront*, Addr> wavefrontLastPC;

    // 4. Work Tracking
    // calculate "Sensitivity" = Delta Instructions / Delta Frequency.
    // This map stores the instruction count from the *previous* check to compute Delta.
    std::map<ComputeUnit*, double> lastCuInstCount;
        
    //5. Polling rate- helps to control polling rate for DVFS
    Tick nextPollTick; 

    // 6. Work Tracking on Wavefront level
    // Stores the instruction count of a specific wavefront from the LAST poll.
    // Key: Wavefront ID (or pointer), Value: Inst Count
    std::map<Wavefront*, double> lastWfInstCount;
    std::map<Wavefront*, double> lastWfSchCycles;
    std::map<Wavefront*, double> lastWfSchStalls;

    // 7. Stores the calculate Sensitivity for reporting 
    std::map<ComputeUnit*, double> currentCuSensitivity; 
    double currentDomainSensitivity;
   
    // 8. adding variables for autotuning of threshold
    double globalMaxSensitivity = 1.0; // Default to 1.0 (conservative start)
    const double DECAY_FACTOR = 0.95; // Slowly forget old peaks

    //---------------------------------------------------------------------------------------------

    // --- STANDARD GEM5 MEMBERS ---
    typedef std::map<DomainID, SrcClockDomain *> Domains;
    Domains domains;
    
    SrcClockDomain *sysClkDomain;
    bool enableHandler;
    Tick _transLatency;

    // Pointer to the real GPU hardware
    Shader *gpuShader;
    
    EventFunctionWrapper decisionEvent;

    
    // ----------------------------------------------------------------------
    // Core Logic Functions
    // ----------------------------------------------------------------------

    void runDecisionLoop();
    int checkIfGPUIsRunning();
    int dumpImportantStatsToConsole();
    SrcClockDomain *findDomain(DomainID domain_id) const;

    /**
     * UpdateEvent
     * A specialized event that performs the actual physical clock change.
     * Separate the "Decision" (logic) from the "Update" (actuation) to 
     * allow for modeling transition latency if desired.
     */
    struct UpdateEvent : public Event
    {
        GpuDVFSHandler *handler;
        DomainID domainIDToSet;
        PerfLevel perfLevelToSet;

        UpdateEvent() : Event(Default_Pri, AutoDelete), handler(nullptr) {}
        
        void process() override { updatePerfLevel(); }
        void updatePerfLevel();
        const char *description() const override { return "GPU DVFS Update"; }
    };
};

} // namespace gem5

#endif // __SIM_GPU_DVFS_HANDLER_HH__


