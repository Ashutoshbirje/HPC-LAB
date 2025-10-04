
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

using State = string; // 9 chars '0'..'8' where '0' is blank

const vector<int> dr = {-1, 1, 0, 0};
const vector<int> dc = {0, 0, -1, 1};

vector<State> expand_state(const State &s) {
    vector<State> res;
    int idx = s.find('0');
    int r = idx / 3, c = idx % 3;
    for (int k = 0; k < 4; ++k) {
        int nr = r + dr[k], nc = c + dc[k];
        if (nr >= 0 && nr < 3 && nc >= 0 && nc < 3) {
            int nidx = nr * 3 + nc;
            State t = s;
            swap(t[idx], t[nidx]);
            res.push_back(t);
        }
    }
    return res;
}

int bfs_sequential(const State &start, const State &goal) {
    if (start == goal) return 0;
    unordered_set<State> vis;
    queue<pair<State,int>> q;
    vis.reserve(100000);
    vis.insert(start);
    q.push({start,0});
    while(!q.empty()){
        auto [cur, d] = q.front(); q.pop();
        auto kids = expand_state(cur);
        for(auto &nx : kids){
            if(vis.find(nx) != vis.end()) continue;
            if(nx == goal) return d+1;
            vis.insert(nx);
            q.push({nx, d+1});
        }
    }
    return -1;
}

// Parallel level-by-level BFS
int bfs_parallel(const State &start, const State &goal, int num_threads) {
    if (start == goal) return 0;
    unordered_set<State> visited;
    visited.reserve(200000);
    visited.insert(start);
    vector<State> frontier;
    frontier.push_back(start);
    int depth = 0;

    omp_set_num_threads(num_threads);

    while(!frontier.empty()){
        vector<State> next_level;
        // Reserve space heuristically
        next_level.reserve(frontier.size() * 2);

        // Parallel expansion of nodes in frontier
        #pragma omp parallel
        {
            vector<State> local_next; // thread-local
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < (int)frontier.size(); ++i) {
                const State &cur = frontier[i];
                auto kids = expand_state(cur);
                for (auto &nx : kids) {
                    // we'll test visited in a thread-safe manner
                    bool do_add = false;
                    #pragma omp critical
                    {
                        if (visited.find(nx) == visited.end()) {
                            visited.insert(nx);
                            do_add = true;
                        }
                    }
                    if (do_add) {
                        if (nx == goal) {
                            // Found; we need to ensure other threads can stop early.
                            // We'll push to local_next and then detect goal later.
                            local_next.push_back(nx);
                        } else {
                            local_next.push_back(nx);
                        }
                    }
                }
            }
            // merge local_next into next_level
            if (!local_next.empty()) {
                #pragma omp critical
                {
                    next_level.insert(next_level.end(), local_next.begin(), local_next.end());
                }
            }
        } // end parallel

        ++depth;
        // Check if goal is in next_level
        for (const auto &s : next_level) {
            if (s == goal) return depth;
        }
        frontier.swap(next_level);
    }
    return -1;
}

// Helper to print a state
void print_state(const State &s) {
    for (int i = 0; i < 9; ++i) {
        if (i % 3 == 0) cout << '\n';
        cout << (s[i] == '0' ? '.' : s[i]) << ' ';
    }
    cout << '\n';
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Example start and goal (solvable config)
    // Start: a scrambled state
    State start = "281043765"; // sample
    State goal  = "123456780"; // canonical goal (0 as blank)

    cout << "8-Puzzle Exploratory (BFS)\n";
    cout << "Start state:";
    print_state(start);
    cout << "Goal state:";
    print_state(goal);
    cout << "\n";

    // Sequential
    double t0 = omp_get_wtime();
    int depth_seq = bfs_sequential(start, goal);
    double t1 = omp_get_wtime();
    double time_seq = (t1 - t0) * 1000.0; // ms

    cout << "Sequential BFS result depth: " << depth_seq << ", time: " << time_seq << " ms\n";

    // Parallel (choose thread counts 2,4,8)
    vector<int> threads_to_try = {2, 4, 8};
    for (int threads : threads_to_try) {
        double tp0 = omp_get_wtime();
        int depth_par = bfs_parallel(start, goal, threads);
        double tp1 = omp_get_wtime();
        double time_par = (tp1 - tp0) * 1000.0;
        cout << "Parallel BFS (" << threads << " threads) depth: " << depth_par
             << ", time: " << time_par << " ms";
        if (time_par > 0.0) {
            cout << ", speedup: " << fixed << setprecision(2) << (time_seq / time_par) << "x";
        }
        cout << "\n";
    }

    cout << "\nNotes: This simple parallelization parallelizes expansions within a level. For more advanced work-stealing/tasking, OpenMP tasks or MPI-based partitioning can be used.\n";
    return 0;
}
