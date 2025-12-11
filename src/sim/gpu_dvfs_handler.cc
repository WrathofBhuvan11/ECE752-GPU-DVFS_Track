#include "sim/gpu_dvfs_handler.hh"
#include "base/trace.hh"
#include "debug/GpuDVFS.hh"
#include "sim/stat_control.hh"
#include "gpu-compute/shader.hh"
#include "gpu-compute/compute_unit.hh"
#include <cmath>
#include <limits>
#include "sim/core.hh"

namespace gem5
{

// ----------------------------------------------------------------------
// CONSTANTS FOR TUNING
// ----------------------------------------------------------------------
// Target IPC to consider a wave "Compute Bound". 
// 0.08 is a reasonable target for a healthy vector workload.
const double TARGET_WAVE_IPC = 0.08; 

// Base Performance for Memory-Bound code (Sensitivity = 0).
// This prevents "Divide by Zero" in ED2P and represents DRAM throughput.
const double BASE_PERF_MEM = 0.2; 

GpuDVFSHandler::GpuDVFSHandler(const Params &p)
    : SimObject(p),
      sysClkDomain(p.sys_clk_domain),
      enableHandler(p.enable),
      _transLatency(p.transition_latency),
      pollingInterval(p.polling_interval),
      gpuShader(dynamic_cast<Shader *>(p.shader)),
      decisionEvent([this]{ runDecisionLoop(); }, name())
{
    int cuId = 0;
    for (auto *cu : gpuShader->cuList) {
        DomainID did = cuId;
        if (!p.domains.empty()) {
            domains[did] = p.domains[cuId % p.domains.size()];
        }
        cuToDomain[cu] = did;
        cuIdMap[cu] = cuId++;
    }
}

void GpuDVFSHandler::startup()
{
    if (enableHandler) {
        // Start neutrally
        for (int i = 0; i < TABLE_SIZE; i++) sensitivityTable[i] = 0.5;

        wavefrontLastPC.clear();
        wfCreationTick.clear();
        lastWfInstCount.clear();
        lastWfSchCycles.clear();
        lastWfSchStalls.clear();

        schedule(decisionEvent, curTick() + 5000000000); 
    }
}

SrcClockDomain *
GpuDVFSHandler::findDomain(DomainID domain_id) const
{
    auto it = domains.find(domain_id);
    return (it != domains.end()) ? it->second : nullptr;
}

int GpuDVFSHandler::checkIfGPUIsRunning()
{
    if (!gpuShader) return 0;
    for (auto *cu : gpuShader->cuList) {
        for (const auto &simd_waves : cu->wfList) {
            for (auto *wave : simd_waves) {
                if (wave->getStatus() != Wavefront::S_STOPPED) return 1;
            }
        }
    }
    return 0;
}

// --------------------------------------------------------------------------
// 1. SENSITIVITY MODEL
// --------------------------------------------------------------------------
double GpuDVFSHandler::computeSensitivity(Wavefront* wf, double deltaSchCycles, double deltaSchmemStalls)
{
    // 1. Safety Check
    if (deltaSchCycles <= 0.0) return 0.0; 

    // 2. The Formula: S = 1.0 - (MemoryStalls / TotalCycles)
    // deltaSchmemStalls- the MEMORY stalls passed in
    double memBoundRatio = deltaSchmemStalls / deltaSchCycles;
    
    double S = 1.0 - memBoundRatio;

    // 3. Clamp
    if (S < 0.0) S = 0.0;
    if (S > 1.0) S = 1.0;

    return S;
}

// --------------------------------------------------------------------------
// 2. PERFORMANCE & POWER MODELS
// --------------------------------------------------------------------------
double GpuDVFSHandler::predictPerf(double S, double fMHz, double fNomMHz)
{
    // If S=1, Perf scales linearly with Freq.
    // If S=0, Perf is fixed at BASE_PERF_MEM.
    double fNorm = fMHz / fNomMHz;
    return BASE_PERF_MEM + (S * fNorm);
}

double GpuDVFSHandler::computePower(double fMHz, double voltage)
{
    // Power = C * V^2 * f
    double dynamic = C_DYNAMIC * voltage * voltage * (fMHz / 1000.0) * A_ACTIVITY; 
    return dynamic;
}

double GpuDVFSHandler::computeEDP(double perf, double power)
{
    if (perf <= 1e-6) return 1e15; // Penalty for zero perf
    // Cost = Power / Perf^2
    return power / (perf * perf);
}
double GpuDVFSHandler::computeED2P(double perf, double power)
{
    if (perf <= 1e-6) return 1e15; // Penalty for zero perf
    // Cost = Power / Perf^3
    return power / (perf * perf * perf);
}

// --------------------------------------------------------------------------
// 3. DECISION LOGIC (The "Threshold")
// --------------------------------------------------------------------------
SrcClockDomain::PerfLevel GpuDVFSHandler::chooseBestLevel(double sensitivity, int cuID) {
    SrcClockDomain *d = domains[cuID];
    int numLevels = d->numPerfLevels();
    
    double bestMetric = std::numeric_limits<double>::max();
    int bestLevel = 0;

    // Check every available frequency level
    for (int i = 0; i < numLevels; i++) {
        // 1. Get Freq (MHz) and Voltage
        double fMHz = (1.0 / d->clkPeriodAtPerfLevel(i)) * 1.0e6;
        double voltage = volts[i];

        // 2. Predict Performance (Linear Scaling Model)
        // If S=1 (Compute), Perf scales with Freq. If S=0 (Mem), it stays flat.
        double predictedPerf = predictPerf(sensitivity, fMHz, 2000.0);

        // 3. Predict Power (P = C*V^2*f)
        double predictedPower = computePower(fMHz, voltage);

        // 4. Calculate Energy-Delay^2 Product (ED2P)
        // MINIMIZE this value
        // double metric = computeED2P(predictedPerf, predictedPower);
        double metric = computeEDP(predictedPerf, predictedPower);

        if (metric < bestMetric) {
            bestMetric = metric;
            bestLevel = i;
        }
    }
    return bestLevel;
}

// --------------------------------------------------------------------------
// MAIN LOOP
// --------------------------------------------------------------------------
void GpuDVFSHandler::runDecisionLoop() {
    // [Check Running]
    if (!checkIfGPUIsRunning()) {
        schedule(decisionEvent, curTick() + pollingInterval);
        return;
    }

    int cu_idx = 0;
    for (auto *cu : gpuShader->cuList) {
        
        // Local Aggregators for this CU
        double cuPredictedSensitivity = 0.0;
        int activeWaves = 0;
        double clockPeriod = cu->clockPeriod();

        // wfList- vector of vectors: wfList[simdId][waveId]
        for (const auto &simd_waves : cu->wfList) {            
        for (auto *w : simd_waves) {
            if (w->status == Wavefront::S_STOPPED) continue;
            //--------------------------------------------------------------
            // 1: MEASURE & VRF (Hardware Snooping)
            //--------------------------------------------------------------
            // Get raw hardware stats and calculate deltas for this epoch.
            // This is the "Ground Truth" from VRF vector register file
            
            // Time Delta (Total Cycles Elapsed)
            Tick currTickVal = curTick();
            Tick lastTickVal = prevWfTick[w]; 
            if (lastTickVal == 0) lastTickVal = currTickVal - pollingInterval;
            
            double deltaCycles = (double)(currTickVal - lastTickVal) / clockPeriod;
            prevWfTick[w] = currTickVal;

            // Stall Delta (Memory Stalls from VRF Scoreboard)
            // VRF Scoreboard- this one not in the paper - newly added
            uint64_t currMemStalls = w->dvfsStats.numMemoryStalls;
            uint64_t lastMemStalls = prevMemStalls[w];
            double deltaMemStalls = (double)(currMemStalls - lastMemStalls);
            prevMemStalls[w] = currMemStalls;

            //--------------------------------------------------------------
            // 2: CALCULATE SENSITIVITY
            //--------------------------------------------------------------
            // Convert raw hardware counters into a 0.0-1.0 metric.
            // S=1.0 (Compute Bound), S=0.0 (Memory Bound).
            double measuredS = computeSensitivity(w, deltaCycles, deltaMemStalls);

            //--------------------------------------------------------------
            // 3: TRAIN (Update PCSTALL Table)
            //--------------------------------------------------------------
            // Map the instruction PC to the sensitivity just measured.
            // This "teaches" the predictor how this code block behaves.
            Addr pc = w->pc();
            int idx = getIndex(pc);

            //EMA Based- Exponential moving average- (1-p)* previous value + p* present value
            sensitivityTable[idx] = (0.6 * measuredS) + (0.4 * sensitivityTable[idx]);

            //--------------------------------------------------------------
            // 4: PREDICT (Accumulate)
            //--------------------------------------------------------------
            // Use the *Learned Table Value* for the decision, not the raw noise.
            // This ensures stability even if one epoch is weird.
            cuPredictedSensitivity += sensitivityTable[idx];
            activeWaves++;
        } 
        } 
        //--------------------------------------------------------------
        // 5: MAKE DECESION & NORMALIZATION
        //--------------------------------------------------------------
        if (activeWaves > 0) {
            // Optimization: Find freq that minimizes EDP based on prediction
            // It can be changed to ED^2P as well
            // Normalization: Summed average - ( chooseBestLevel-> predictPerf ) expects a 
            // Sensitievity S between 0.0 and 1.0; Pure Memory Bound = 0 & Pure Compute Bound = 1;

            PerfLevel nextLevel = chooseBestLevel(cuPredictedSensitivity/activeWaves, cu_idx);
            
            // Schedule the hardware transition
            UpdateEvent *e = new UpdateEvent();
            e->handler = this;
            e->domainIDToSet = cu_idx;
            e->perfLevelToSet = nextLevel;
            schedule(e, curTick() + _transLatency);
        }
        cu_idx++;
    }

    // Logging & Next Loop
    dumpImportantStatsToConsole();
    schedule(decisionEvent, curTick() + pollingInterval);
}

void GpuDVFSHandler::UpdateEvent::updatePerfLevel()
{
    auto d = handler->findDomain(domainIDToSet);
    if (d) d->perfLevel(perfLevelToSet);
}

// --------------------------------------------------------------------------
// STATS DUMP
// --------------------------------------------------------------------------
int GpuDVFSHandler::dumpImportantStatsToConsole()
{
    static double prevInstTotal[64] = {0};
    static double prevNumCycles[64] = {0};
    static double edpTotal = 0.0;
    static double ed2pTotal = 0.0;
    static Tick initTime = 0;

    for (auto *cu : gpuShader->cuList) {
        int cu_idx = cuIdMap[cu];
        
        // Skip idle reporting
        if (cu->stats.totalCycles.total() == prevNumCycles[cu_idx]) continue;

        double ipc = cu->stats.ipc.total();

        if(!std::isnan(ipc)){
            double instr = cu->stats.numInstrExecuted.total();
            double deltaInstr = instr - prevInstTotal[cu_idx];
            if(deltaInstr < 0) deltaInstr = 0;

            double numCycles = cu->stats.totalCycles.total();
            double deltaNumCycles = numCycles - prevNumCycles[cu_idx];
            if(deltaNumCycles <= 0) deltaNumCycles = 1;
            
            double deltaIPC = deltaInstr / deltaNumCycles;
            double voltage = cu->voltage();
            double fHz = cu->frequency();
            double fMHz = fHz / 1e6;

            double S = 0.0;
            if (currentCuSensitivity.find(cu) != currentCuSensitivity.end()) {
                S = currentCuSensitivity[cu];
            }

            double perf = predictPerf(S, fMHz); 
            double power = computePower(fMHz, voltage);
            double edp = computeEDP(perf, power); 
            double ed2p = computeED2P(perf, power);

            edpTotal += edp;
            ed2pTotal += ed2p;

            if (initTime == 0) initTime = curTick();

            inform("GPU_DVFS_STATS: CU: %d, clock: %lld, IPC: %.4f, Freq: %.0f MHz, Voltage: %.2f, EDP: %.2f, ED2P: %.2f, EDP_Total: %.2f, ED2P_Total: %.2f, Sens: %.4f"
               , cu_idx
               , curTick() - initTime
               , deltaIPC
               , fMHz
               , voltage
               , edp
               , ed2p
               , edpTotal
               , ed2pTotal
               , S 
            );

            prevInstTotal[cu_idx] = instr;
            prevNumCycles[cu_idx] = numCycles;
        }
    }
    return 0;
}

double GpuDVFSHandler::normalizeByPriority(Wavefront* wf, const std::vector<Wavefront*>& simd_waves) {
    // return 1.0;
    // Simple Age-Based Weighting
    // Find the oldest wave in the list
    Tick oldestTick = curTick();
    for (auto *w : simd_waves) {
        if (wfCreationTick[w] < oldestTick) oldestTick = wfCreationTick[w];
    }
    // If this is the oldest wave, weight = 1.0 (Full Impact)
    // If younger, weight = 0.5 (Reduced Impact, as it can hide behind the oldest)
    if (wfCreationTick[wf] == oldestTick) return 1.0;
    else return 0.5; 
    // #TODO Tuning parameter required here
}

} // namespace gem5