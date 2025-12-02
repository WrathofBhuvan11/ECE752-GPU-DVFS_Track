#ifndef __SIM_GPU_DVFS_HANDLER_HH__
#define __SIM_GPU_DVFS_HANDLER_HH__

#include "params/GpuDVFSHandler.hh"
#include "sim/sim_object.hh"
#include "sim/clock_domain.hh"
#include "sim/eventq.hh"
#include <map>
//GPU Headers required to spy on Wavefronts
#include "gpu-compute/shader.hh"
#include "gpu-compute/compute_unit.hh"
#include "gpu-compute/wavefront.hh"

namespace gem5
{

/**
 * GpuDVFSHandler
 * handler for managing GPU Dynamic Voltage and Frequency Scaling (DVFS).
 * this class is "active": it contains its own decision loop that wakes up 
 * periodically to inspect GPU state (Program Counter)
 * and make frequency scaling decisions autonomously.
 */
class GpuDVFSHandler : public SimObject
{
  public:
    typedef GpuDVFSHandlerParams Params;
    GpuDVFSHandler(const Params &p);

    // Standard gem5 typedefs for Domain IDs and Performance Levels
    typedef SrcClockDomain::DomainID DomainID;
    typedef SrcClockDomain::PerfLevel PerfLevel;

    /**
     * startup()
     * Called by gem5 after all objects are created but before simulation starts.
     * use this to schedule the first iteration of decision loop.
     */
    void startup() override;

  private:
    /**
     * Container to store pointers to the clock domains this handler manages.
     * Mapped by DomainID (integer) -> SrcClockDomain* (object pointer).
     */
    typedef std::map<DomainID, SrcClockDomain*> Domains;
    Domains domains;
    
    // Pointer to the system clock domain (required for scheduling reference, though unused in toy logic)
    SrcClockDomain *sysClkDomain;
    
    // Master switch to enable/disable the handler from Python config
    bool enableHandler;
    
    // Latency to apply when switching frequencies (simulates PLL lock time)
    Tick _transLatency;

    // Pointer to the real GPU hardware
    Shader *gpuShader;

    /**
     * The main event wrapper. This wraps the 'runDecisionLoop' function
     * so it can be scheduled on the gem5 event queue.
     */
    EventFunctionWrapper decisionEvent;
    
    // ----------------------------------------------------------------------
    // Core Logic Functions
    // ----------------------------------------------------------------------

    /**
     * runDecisionLoop()
     * The "Governor" logic. This function:
     * 1. Wakes up periodically.
     * 2. Reads the GPU PC.
     * 3. Decides the target performance level based on a Modulo-150 policy.  #TODO We will insert PCSTALL Here
     * 4. Schedules an update if necessary.
     * 5. Reschedules itself to run again.
     */
    void runDecisionLoop();

    /**
     * readGpuPC()
     * A helper function to fetch the current Program Counter (PC) from the GPU.
     */
    Addr readGpuPC(); 

    /**
     * findDomain()
     * Helper to retrieve a clock domain object given its ID.
     */
    SrcClockDomain *findDomain(DomainID domain_id) const;

    /**
     * UpdateEvent
     * A specialized event that performs the actual physical clock change.
     * We separate the "Decision" (logic) from the "Update" (actuation) to 
     * allow for modeling transition latency if desired.
     */
    struct UpdateEvent : public Event
    {
        GpuDVFSHandler *handler;       // Pointer back to the parent handler
        DomainID domainIDToSet;        // Which domain to change
        PerfLevel perfLevelToSet;      // Which level (0, 1, 2) to switch to

        UpdateEvent() : Event(Default_Pri, AutoDelete), handler(nullptr) {}
        
        // The process() method is called by the event queue when the event fires
        void process() override { updatePerfLevel(); }
        
        // Performs the actual frequency/voltage switch
        void updatePerfLevel();
        
        const char *description() const override { return "GPU DVFS Update Perf Level"; }
    };
};

} // namespace gem5

#endif // __SIM_GPU_DVFS_HANDLER_HH__
