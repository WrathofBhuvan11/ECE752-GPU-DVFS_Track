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
 * GpuDVFSHandler: PCSTALL-based DVFS for GPUs, per the paper "Predict; Don’t React...".
 * Uses wavefront PC to predict sensitivity, aggregates per CU, minimizes ED²P per CU.
 */
class GpuDVFSHandler : public SimObject
{
  public:
    typedef GpuDVFSHandlerParams Params;
    GpuDVFSHandler(const Params &p);

    typedef SrcClockDomain::DomainID DomainID;
    typedef SrcClockDomain::PerfLevel PerfLevel;

    void startup() override;

  private:
    // PCSTALL Structures
    static const int TABLE_SIZE = 128; // Paper: 128 entries
    double sensitivityTable[TABLE_SIZE]; // Sensitivity per code block

    int getIndex(Addr pc) const { return (pc >> 2) & (TABLE_SIZE - 1); } // 4-instr granularity 

    std::map<Wavefront*, Addr> wavefrontLastPC; // Starting PC for update
    std::map<Wavefront*, Tick> wfCreationTick; // For age-based normalization

    std::map<Wavefront*, double> lastWfInstCount;
    std::map<Wavefront*, double> lastWfSchCycles;
    std::map<Wavefront*, double> lastWfSchStalls;

    std::map<ComputeUnit*, double> currentCuSensitivity; 
    double currentDomainSensitivity; // For stats

    // Per-CU domain mapping (assume sequential IDs)
    std::map<ComputeUnit*, DomainID> cuToDomain;
    std::map<ComputeUnit*, int> cuIdMap;
    std::map<ComputeUnit*, double> lastCuMeasuredSumS;

    // DVFS Levels (3 levels: high/compute, medium, low/memory)
    static const int NUM_LEVELS = 3;
    double freqsMHz[NUM_LEVELS] = {4000, 2000, 1000};
    double volts[NUM_LEVELS] = {1.0, 0.9, 0.8};

    // Map to store the last snapshot of stats for each wavefront
    std::map<Wavefront*, uint64_t> prevTotalCycles;
    std::map<Wavefront*, uint64_t> prevMemStalls;
    std::map<Wavefront*, Tick> prevWfTick;

    // Constants for power model 
    const double C_DYNAMIC = 1.0; // Capacitance factor
    const double A_ACTIVITY = 1.0; // Activity factor

    // Gem5 Members
    typedef std::map<DomainID, SrcClockDomain *> Domains;
    Domains domains;
    SrcClockDomain *sysClkDomain;
    bool enableHandler;
    Tick _transLatency;
    Tick pollingInterval;
    Shader *gpuShader;
    EventFunctionWrapper decisionEvent;

    // Core Functions
    void runDecisionLoop();
    int checkIfGPUIsRunning();
    int dumpImportantStatsToConsole();
    SrcClockDomain *findDomain(DomainID domain_id) const;

    // Metrics Definitions
    double computeSensitivity(Wavefront* wf, double deltaSchCycles, double deltaSchmemStalls);
    double normalizeByPriority(Wavefront* wf, const std::vector<Wavefront*>& simd_waves);
    double predictPerf(double S, double fMHz, double fNomMHz = 2000.0); // perf(f) = perf_base + S * (f / f_nom)
    double computePower(double fMHz, double v); // P = C * V^2 * f * A + leakage
    double computeEDP(double perf, double power); // EDP proxy = power * (epochT / perf)^2
    double computeED2P(double perf, double power); // ED2P proxy = power * (epochT / perf)^3

    // Decision Logic 
    PerfLevel chooseBestLevel(double cuSumS, int cuID);

    // Update Event
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
