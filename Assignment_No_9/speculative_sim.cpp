
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using ull = unsigned long long;

std::atomic<bool> stop_flag(false);
std::atomic<int> decided_policy(-1); // -1 undecided, 0 conservative, 1 optimistic

struct SimResult {
    ull work_units; // amount of simulated "work"
    double time_ms;
};

// Simulate CPU-bound "work" for `steps` iterations. Periodically checks stop_flag.
SimResult simulate_model(const string &name, ull steps, int check_every=1000) {
    ull work = 0;
    double t0 = omp_get_wtime();
    ull acc = 0;
    for (ull i = 0; i < steps; ++i) {
        // small dummy computation to consume CPU
        acc += (i * 132491 + 7) % 1009;
        ++work;
        if ((i & (check_every - 1)) == 0) {
            if (stop_flag.load(std::memory_order_relaxed)) {
                break;
            }
        }
    }
    double t1 = omp_get_wtime();
    (void)acc; // prevent compiler optimizing away
    SimResult r;
    r.work_units = work;
    r.time_ms = (t1 - t0) * 1000.0;
    return r;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // parameters
    int decision_ms = 2000;   // how long until decision is revealed (ms)
    ull work_steps = 100000000ULL; // how many work iterations each model would ideally do
    if (argc >= 2) decision_ms = atoi(argv[1]);
    if (argc >= 3) work_steps = strtoull(argv[2], nullptr, 10);

    cout << "Speculative Simulation\n";
    cout << "Decision after (ms): " << decision_ms << ", per-model work_steps: " << work_steps << "\n";

    // Seed decision with time-based randomness
    std::mt19937 rng((unsigned)time(nullptr));
    std::uniform_int_distribution<int> dist_policy(0,1);

    // Parallel speculative run: run both concurrently, decision after decision_ms sets which to keep
    stop_flag.store(false);
    decided_policy.store(-1);
    SimResult res_cons{0,0}, res_opt{0,0};
    double start_all = omp_get_wtime();

    #pragma omp parallel sections num_threads(3)
    {
        #pragma omp section
        {
            // Conservative model
            res_cons = simulate_model("Conservative", work_steps);
        }
        #pragma omp section
        {
            // Optimistic model
            res_opt = simulate_model("Optimistic", work_steps);
        }
        #pragma omp section
        {
            // Decision thread: wait decision_ms then decide which policy to keep
            double t0 = omp_get_wtime();
            // busy wait with sleep to be friendly to CPU
            int slept = 0;
            while (true) {
                double now = omp_get_wtime();
                if ((now - t0) * 1000.0 >= decision_ms) break;
                // small sleep
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                ++slept;
            }
            int chosen = dist_policy(rng); // randomly choose 0 or 1
            decided_policy.store(chosen);
            if (chosen == 0) {
                // keep conservative => stop optimistic
                stop_flag.store(true);
                cout << "[Decision] Keeping Conservative model\n";
            } else {
                stop_flag.store(true);
                cout << "[Decision] Keeping Optimistic model\n";
            }
            // done
        }
    } // end parallel sections

    double end_all = omp_get_wtime();
    double parallel_total_ms = (end_all - start_all) * 1000.0;

    int chosen = decided_policy.load();
    // If decision thread didn't run for some reason (shouldn't), default choose 0
    if (chosen == -1) chosen = 0;

    // Determine which result was discarded and calculate wasted work
    ull kept_work = (chosen == 0 ? res_cons.work_units : res_opt.work_units);
    ull discarded_work = (chosen == 0 ? res_opt.work_units : res_cons.work_units);
    ull total_work = res_cons.work_units + res_opt.work_units;
    double wasted_pct = 0.0;
    if (total_work > 0) wasted_pct = (100.0 * (double)discarded_work) / (double)total_work;

    cout << fixed << setprecision(3);
    cout << "Parallel (speculative) total time: " << parallel_total_ms << " ms\n";
    cout << "Conservative model: work_units=" << res_cons.work_units << ", time=" << res_cons.time_ms << " ms\n";
    cout << "Optimistic  model: work_units=" << res_opt.work_units  << ", time=" << res_opt.time_ms  << " ms\n";
    cout << "Chosen model: " << (chosen==0 ? "Conservative" : "Optimistic") << "\n";
    cout << "Wasted computation (discarded model) = " << discarded_work << " units\n";
    cout << "Total work done by both models = " << total_work << " units\n";
    cout << "Wasted computation (%) = " << wasted_pct << "%\n";

    // Sequential baseline: run only the chosen model fully (no speculation)
    double tseq0 = omp_get_wtime();
    SimResult seq_res;
    if (chosen == 0) {
        seq_res = simulate_model("Conservative(sequential)", work_steps);
    } else {
        seq_res = simulate_model("Optimistic(sequential)", work_steps);
    }
    double tseq1 = omp_get_wtime();
    double seq_time_ms = (tseq1 - tseq0) * 1000.0;
    cout << "Sequential (only chosen model) time: " << seq_time_ms << " ms\n";

    // Compare times
    double speedup = seq_time_ms / parallel_total_ms;
    cout << "Speedup (sequential / speculative_parallel): " << speedup << "x\n";

    return 0;
}
