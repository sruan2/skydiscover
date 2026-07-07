#include <bits/stdc++.h>
using namespace std;

// EVOLVE-BLOCK-START
struct Orientation {
    int r = 0, f = 0;
    long long minx = 0, miny = 0;
    long long w = 0, h = 0;
    vector<pair<long long, long long>> cells;
};

struct Piece {
    vector<pair<long long, long long>> cells;
    vector<Orientation> ori;
};

struct Placement {
    long long x = 0, y = 0;
    int r = 0, f = 0;
};

static const long long INF = (long long)4e18;

static long long sat_add(long long a, long long b) {
    if (a >= INF || b >= INF || a > INF - b) return INF;
    return a + b;
}

static long long sat_mul(long long a, long long b) {
    __int128 v = (__int128)a * (__int128)b;
    return v > INF ? INF : (long long)v;
}

static pair<long long, long long> transform_cell(long long x, long long y, int r, int f) {
    if (f) x = -x;
    if (r == 0) return {x, y};
    if (r == 1) return {y, -x};
    if (r == 2) return {-x, -y};
    return {-y, x};
}

static long long ceil_sqrt_ll(long long v) {
    long long r = sqrt((long double)v);
    while (r * r < v) ++r;
    while (r > 0 && (r - 1) * (r - 1) >= v) --r;
    return r;
}

static bool better_orientation(const Orientation& a, const Orientation& b, int mode) {
    long long area_a = sat_mul(a.w, a.h);
    long long area_b = sat_mul(b.w, b.h);
    if (mode == 1) {
        if (area_a != area_b) return area_a < area_b;
        if (max(a.w, a.h) != max(b.w, b.h)) return max(a.w, a.h) < max(b.w, b.h);
        if (a.h != b.h) return a.h < b.h;
        return a.w < b.w;
    }
    if (mode == 2) {
        if (a.w != b.w) return a.w < b.w;
        if (a.h != b.h) return a.h < b.h;
        return area_a < area_b;
    }
    if (mode == 3) {
        long long da = llabs(a.w - a.h), db = llabs(b.w - b.h);
        if (da != db) return da < db;
        if (max(a.w, a.h) != max(b.w, b.h)) return max(a.w, a.h) < max(b.w, b.h);
        return area_a < area_b;
    }
    if (a.h != b.h) return a.h < b.h;
    if (area_a != area_b) return area_a < area_b;
    return a.w < b.w;
}

static int choose_orientation(const Piece& p, long long side, int mode) {
    int best = -1;
    for (int i = 0; i < (int)p.ori.size(); ++i) {
        const Orientation& o = p.ori[i];
        if (o.w > side) continue;
        if (best == -1) {
            best = i;
            continue;
        }
        const Orientation& b = p.ori[best];
        if (better_orientation(o, b, mode)) {
            best = i;
        }
    }
    return best;
}

struct PairHash {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    size_t operator()(const pair<long long, long long>& p) const {
        uint64_t a = splitmix64((uint64_t)p.first);
        uint64_t b = splitmix64((uint64_t)p.second + 0x9e3779b97f4a7c15ULL);
        return (size_t)(a ^ (b << 1));
    }
};

struct XorShift64 {
    uint64_t s;
    explicit XorShift64(uint64_t seed) : s(seed) {}

    uint64_t next() {
        s ^= s << 7;
        s ^= s >> 9;
        return s;
    }

    long long uniform(long long lim) {
        if (lim <= 0) return 0;
        return (long long)(next() % (uint64_t)lim);
    }
};

static bool can_place_exact(
    const Orientation& o,
    long long rx,
    long long ry,
    long long side,
    const unordered_set<pair<long long, long long>, PairHash>& occupied
) {
    if (rx < 0 || ry < 0 || rx + o.w > side || ry + o.h > side) return false;
    for (auto [cx, cy] : o.cells) {
        pair<long long, long long> cell = {rx + cx, ry + cy};
        if (occupied.find(cell) != occupied.end()) return false;
    }
    return true;
}

static void place_exact(
    const Orientation& o,
    long long rx,
    long long ry,
    unordered_set<pair<long long, long long>, PairHash>& occupied,
    vector<pair<long long, long long>>& occ_list
) {
    for (auto [cx, cy] : o.cells) {
        pair<long long, long long> cell = {rx + cx, ry + cy};
        occupied.insert(cell);
        occ_list.push_back(cell);
    }
}

static bool exact_pack_side(
    const vector<Piece>& pieces,
    long long side,
    vector<Placement>& out_place,
    chrono::steady_clock::time_point deadline,
    int strategy
) {
    const int n = (int)pieces.size();
    vector<vector<int>> fit(n);
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < (int)pieces[i].ori.size(); ++j) {
            const Orientation& o = pieces[i].ori[j];
            if (o.w <= side && o.h <= side) fit[i].push_back(j);
        }
        if (fit[i].empty()) return false;
        sort(fit[i].begin(), fit[i].end(), [&](int a, int b) {
            const Orientation& oa = pieces[i].ori[a];
            const Orientation& ob = pieces[i].ori[b];
            long long aa = sat_mul(oa.w, oa.h), ab = sat_mul(ob.w, ob.h);
            if (strategy == 1) {
                long long pa = sat_mul(side - oa.w + 1, side - oa.h + 1);
                long long pb = sat_mul(side - ob.w + 1, side - ob.h + 1);
                if (pa != pb) return pa < pb;
                if (max(oa.w, oa.h) != max(ob.w, ob.h)) return max(oa.w, oa.h) > max(ob.w, ob.h);
                return aa < ab;
            }
            if (strategy == 2) {
                if (oa.h != ob.h) return oa.h > ob.h;
                if (oa.w != ob.w) return oa.w > ob.w;
                return aa < ab;
            }
            if (strategy == 3) {
                long long pa = sat_mul(side - oa.w + 1, side - oa.h + 1);
                long long pb = sat_mul(side - ob.w + 1, side - ob.h + 1);
                if (max(oa.w, oa.h) != max(ob.w, ob.h)) return max(oa.w, oa.h) > max(ob.w, ob.h);
                if (pa != pb) return pa < pb;
                if (llabs(oa.w - oa.h) != llabs(ob.w - ob.h)) return llabs(oa.w - oa.h) < llabs(ob.w - ob.h);
                return aa < ab;
            }
            if (aa != ab) return aa < ab;
            if (max(oa.w, oa.h) != max(ob.w, ob.h)) return max(oa.w, oa.h) < max(ob.w, ob.h);
            if (oa.h != ob.h) return oa.h < ob.h;
            return oa.w < ob.w;
        });
    }

    sort(order.begin(), order.end(), [&](int a, int b) {
        const Orientation& oa = pieces[a].ori[fit[a][0]];
        const Orientation& ob = pieces[b].ori[fit[b][0]];
        long long aa = sat_mul(oa.w, oa.h), ab = sat_mul(ob.w, ob.h);
        long long pa = sat_mul(side - oa.w + 1, side - oa.h + 1);
        long long pb = sat_mul(side - ob.w + 1, side - ob.h + 1);
        if (strategy == 1 && pa != pb) return pa < pb;
        if (strategy == 2 && max(oa.w, oa.h) != max(ob.w, ob.h)) {
            return max(oa.w, oa.h) > max(ob.w, ob.h);
        }
        if (strategy == 3) {
            if (max(oa.w, oa.h) != max(ob.w, ob.h)) return max(oa.w, oa.h) > max(ob.w, ob.h);
            if (pa != pb) return pa < pb;
        }
        if (aa != ab) return aa > ab;
        if (max(oa.w, oa.h) != max(ob.w, ob.h)) return max(oa.w, oa.h) > max(ob.w, ob.h);
        if (pieces[a].cells.size() != pieces[b].cells.size()) return pieces[a].cells.size() > pieces[b].cells.size();
        return a < b;
    });

    vector<Placement> place(n);
    unordered_set<pair<long long, long long>, PairHash> occupied;
    vector<pair<long long, long long>> occ_list;
    vector<pair<long long, long long>> frontier;
    size_t total_cells = 0;
    for (const Piece& p : pieces) total_cells += p.cells.size();
    occupied.reserve(total_cells * 3 + 100);
    occ_list.reserve(total_cells);
    frontier.reserve(min<size_t>(total_cells * 4 + 16, 600000));

    XorShift64 rng(0x123456789abcdefULL ^ (uint64_t)side ^ ((uint64_t)n << 32));
    const long long dirs[5][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int placed_count = 0;

    for (int id : order) {
        if (chrono::steady_clock::now() > deadline) return false;
        bool done = false;

        auto try_origin = [&](int oi, long long rx, long long ry) {
            if (done) return;
            const Orientation& o = pieces[id].ori[oi];
            if (!can_place_exact(o, rx, ry, side, occupied)) return;
            place[id].x = rx - o.minx;
            place[id].y = ry - o.miny;
            place[id].r = o.r;
            place[id].f = o.f;
            place_exact(o, rx, ry, occupied, occ_list);
            if (frontier.size() < 600000) {
                for (auto [cx, cy] : o.cells) {
                    long long px = rx + cx;
                    long long py = ry + cy;
                    for (int d = 1; d < 5; ++d) {
                        long long nx = px + dirs[d][0];
                        long long ny = py + dirs[d][1];
                        if (nx >= 0 && ny >= 0 && nx < side && ny < side) {
                            pair<long long, long long> nb = {nx, ny};
                            if (occupied.find(nb) == occupied.end()) frontier.push_back(nb);
                        }
                    }
                }
            }
            done = true;
        };

        for (int oi : fit[id]) {
            const Orientation& o = pieces[id].ori[oi];
            if (strategy == 3) {
                long long cx[4] = {0, side - o.w, side - o.w, 0};
                long long cy[4] = {0, 0, side - o.h, side - o.h};
                int start = placed_count & 3;
                for (int t = 0; t < 4; ++t) {
                    int c = (start + t) & 3;
                    try_origin(oi, cx[c], cy[c]);
                    if (done) break;
                }
            } else if (strategy == 2) {
                try_origin(oi, side - o.w, side - o.h);
                try_origin(oi, 0, side - o.h);
                try_origin(oi, side - o.w, 0);
                try_origin(oi, 0, 0);
            } else {
                try_origin(oi, 0, 0);
                try_origin(oi, side - o.w, 0);
                try_origin(oi, 0, side - o.h);
                try_origin(oi, side - o.w, side - o.h);
            }
            if (done) break;
        }

        if (strategy == 3 && !done) {
            for (int oi : fit[id]) {
                const Orientation& o = pieces[id].ori[oi];
                long long mx = (side - o.w) / 2;
                long long my = (side - o.h) / 2;
                try_origin(oi, mx, 0);
                try_origin(oi, mx, side - o.h);
                try_origin(oi, 0, my);
                try_origin(oi, side - o.w, my);
                if (done) break;
            }
        }

        if (strategy == 3 && !done) {
            for (int oi : fit[id]) {
                const Orientation& o = pieces[id].ori[oi];
                long long maxx = side - o.w;
                long long maxy = side - o.h;
                try_origin(oi, maxx / 2, maxy / 2);
                try_origin(oi, maxx / 4, maxy / 4);
                try_origin(oi, (maxx * 3) / 4, maxy / 4);
                try_origin(oi, maxx / 4, (maxy * 3) / 4);
                try_origin(oi, (maxx * 3) / 4, (maxy * 3) / 4);
                if (done) break;
            }
        }

        int frontier_cap = n > 5000 ? 48 : (n > 1500 ? 80 : 160);
        int frontier_checks = 0;
        for (int pass = 0; pass < 2 && !done && !frontier.empty(); ++pass) {
            int seen = 0;
            for (long long idx = (long long)frontier.size() - 1; idx >= 0 && !done; --idx) {
                ++seen;
                if (pass == 0 && seen > frontier_cap) break;
                if (pass == 1 && seen > frontier_cap * 80) break;
                if (pass == 1 && (idx % 17 != (long long)(id % 17))) continue;
                auto fc = frontier[(size_t)idx];
                if (occupied.find(fc) != occupied.end()) continue;
                for (int oi : fit[id]) {
                    const Orientation& o = pieces[id].ori[oi];
                    for (auto pc : o.cells) {
                        if (frontier_checks++ >= frontier_cap * 3) break;
                        try_origin(oi, fc.first - pc.first, fc.second - pc.second);
                        if (done) break;
                    }
                    if (done || frontier_checks >= frontier_cap * 3) break;
                }
                if (pass == 1 && frontier_checks >= frontier_cap * 3) break;
            }
        }

        int anchor_trials = occ_list.empty() ? 0 : 48;
        for (int t = 0; t < anchor_trials && !done; ++t) {
            auto occ = occ_list[(size_t)rng.uniform((long long)occ_list.size())];
            for (int oi : fit[id]) {
                const Orientation& o = pieces[id].ori[oi];
                auto pc = o.cells[(size_t)rng.uniform((long long)o.cells.size())];
                for (int d = 1; d < 5; ++d) {
                    long long rx = occ.first + dirs[d][0] - pc.first;
                    long long ry = occ.second + dirs[d][1] - pc.second;
                    try_origin(oi, rx, ry);
                    if (done) break;
                }
                if (done) break;
            }
        }

        int random_trials = 64;
        for (int t = 0; t < random_trials && !done; ++t) {
            for (int oi : fit[id]) {
                const Orientation& o = pieces[id].ori[oi];
                long long rx = rng.uniform(side - o.w + 1);
                long long ry = rng.uniform(side - o.h + 1);
                try_origin(oi, rx, ry);
                if (done) break;
            }
        }

        if (!done && side <= 450 && n <= 800) {
            long long checked = 0;
            for (long long ry = 0; ry < side && !done && checked < 6000; ++ry) {
                for (long long rx = 0; rx < side && !done && checked < 6000; ++rx, ++checked) {
                    for (int oi : fit[id]) {
                        try_origin(oi, rx, ry);
                        if (done) break;
                    }
                }
            }
        }

        if (!done) return false;
        ++placed_count;
    }

    out_place.swap(place);
    return true;
}

static bool pack_with_side(
    const vector<Piece>& pieces,
    long long side,
    int mode,
    long long& out_side,
    vector<Placement>& out_place
) {
    const int n = (int)pieces.size();
    vector<int> chosen(n);
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);

    for (int i = 0; i < n; ++i) {
        chosen[i] = choose_orientation(pieces[i], side, mode);
        if (chosen[i] < 0) return false;
    }

    sort(order.begin(), order.end(), [&](int a, int b) {
        const Orientation& oa = pieces[a].ori[chosen[a]];
        const Orientation& ob = pieces[b].ori[chosen[b]];
        if (oa.h != ob.h) return oa.h > ob.h;
        if (oa.w != ob.w) return oa.w > ob.w;
        return a < b;
    });

    vector<Placement> place(n);
    struct Shelf {
        long long y = 0, h = 0, used = 0;
    };
    vector<Shelf> shelves;
    set<pair<long long, int>> open_shelves;
    long long used_h = 0;

    for (int id : order) {
        const Orientation& o = pieces[id].ori[chosen[id]];
        int best_shelf = -1;
        auto it = open_shelves.upper_bound({side - o.w, INT_MAX});
        if (it != open_shelves.begin()) {
            --it;
            best_shelf = it->second;
            open_shelves.erase(it);
        }

        if (best_shelf == -1) {
            Shelf sh;
            sh.y = used_h;
            sh.h = o.h;
            shelves.push_back(sh);
            best_shelf = (int)shelves.size() - 1;
            used_h = sat_add(used_h, o.h);
        }

        Shelf& sh = shelves[best_shelf];
        place[id].x = sh.used - o.minx;
        place[id].y = sh.y - o.miny;
        place[id].r = o.r;
        place[id].f = o.f;
        sh.used = sat_add(sh.used, o.w);
        if (sh.used < side) {
            open_shelves.insert({sh.used, best_shelf});
        }
    }

    out_side = max(side, used_h);
    out_place.swap(place);
    return true;
}

int main() {
    auto start_time = chrono::steady_clock::now();
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<Piece> pieces(n);
    long long total_cells = 0;
    long long total_min_box_area = 0;
    long long max_min_width = 1;
    long long single_row_side = 0;

    for (int i = 0; i < n; ++i) {
        int k;
        cin >> k;
        total_cells += k;
        pieces[i].cells.resize(k);
        for (int j = 0; j < k; ++j) {
            cin >> pieces[i].cells[j].first >> pieces[i].cells[j].second;
        }

        set<vector<pair<long long, long long>>> seen_shapes;
        for (int f = 0; f <= 1; ++f) {
            for (int r = 0; r < 4; ++r) {
                long long minx = LLONG_MAX, miny = LLONG_MAX;
                long long maxx = LLONG_MIN, maxy = LLONG_MIN;
                vector<pair<long long, long long>> transformed;
                transformed.reserve(pieces[i].cells.size());
                for (auto [cx, cy] : pieces[i].cells) {
                    auto [tx, ty] = transform_cell(cx, cy, r, f);
                    transformed.push_back({tx, ty});
                    minx = min(minx, tx);
                    miny = min(miny, ty);
                    maxx = max(maxx, tx);
                    maxy = max(maxy, ty);
                }
                Orientation o;
                o.r = r;
                o.f = f;
                o.minx = minx;
                o.miny = miny;
                o.w = maxx - minx + 1;
                o.h = maxy - miny + 1;
                o.cells.reserve(transformed.size());
                for (auto [tx, ty] : transformed) {
                    o.cells.push_back({tx - minx, ty - miny});
                }
                sort(o.cells.begin(), o.cells.end());
                if (seen_shapes.insert(o.cells).second) {
                    pieces[i].ori.push_back(o);
                }
            }
        }

        long long best_area = LLONG_MAX;
        long long best_width = LLONG_MAX;
        long long best_flat_h = LLONG_MAX;
        long long best_flat_w = LLONG_MAX;
        for (const Orientation& o : pieces[i].ori) {
            long long area = sat_mul(o.w, o.h);
            best_area = min(best_area, area);
            best_width = min(best_width, o.w);
            if (o.h < best_flat_h || (o.h == best_flat_h && o.w < best_flat_w)) {
                best_flat_h = o.h;
                best_flat_w = o.w;
            }
        }
        total_min_box_area = sat_add(total_min_box_area, best_area);
        max_min_width = max(max_min_width, best_width);
        single_row_side = sat_add(single_row_side, best_flat_w);
    }

    long long lower = max({1LL, ceil_sqrt_ll(total_cells), max_min_width});
    long long center = max(lower, ceil_sqrt_ll(max(total_cells, total_min_box_area)));

    vector<long long> candidates;
    auto add_candidate = [&](long long v) {
        if (v >= lower && v <= single_row_side) candidates.push_back(v);
    };

    add_candidate(lower);
    add_candidate(center);
    for (long long num = 7; num <= 22; ++num) add_candidate(max(1LL, sat_mul(center, num) / 10));
    for (long long v = lower; v <= single_row_side && v - lower <= 160; ++v) add_candidate(v);

    long long v = lower;
    while (v <= single_row_side) {
        add_candidate(v);
        long long step = max(1LL, v / 20);
        if (v > INF - step) break;
        long long nv = v + step;
        v = nv;
    }
    add_candidate(single_row_side);

    sort(candidates.begin(), candidates.end());
    candidates.erase(unique(candidates.begin(), candidates.end()), candidates.end());

    long long best_side = LLONG_MAX;
    vector<Placement> best_place;
    for (long long cand : candidates) {
        for (int mode = 0; mode < 4; ++mode) {
            long long packed_side = 0;
            vector<Placement> place;
            if (!pack_with_side(pieces, cand, mode, packed_side, place)) continue;
            if (packed_side < best_side) {
                best_side = packed_side;
                best_place.swap(place);
            }
        }
    }

    if (best_place.empty()) {
        pack_with_side(pieces, single_row_side, 0, best_side, best_place);
    }

    long long max_min_dim = 1;
    for (const Piece& p : pieces) {
        long long best_dim = LLONG_MAX;
        for (const Orientation& o : p.ori) {
            best_dim = min(best_dim, max(o.w, o.h));
        }
        max_min_dim = max(max_min_dim, best_dim);
    }

    long long exact_lower = max(ceil_sqrt_ll(total_cells), max_min_dim);
    if (exact_lower < best_side) {
        vector<long long> exact_candidates;
        auto add_exact = [&](long long s) {
            if (s >= exact_lower && s < best_side) exact_candidates.push_back(s);
        };

        add_exact(exact_lower);
        for (long long num = 105; num <= 160; num += 5) {
            add_exact(max(exact_lower, sat_mul(exact_lower, num) / 100));
        }
        for (long long delta = 1; delta <= 24; ++delta) add_exact(exact_lower + delta);
        for (long long delta = 1; delta <= 80; delta += 4) add_exact(best_side - delta);
        add_exact((exact_lower + best_side) / 2);

        sort(exact_candidates.begin(), exact_candidates.end());
        exact_candidates.erase(unique(exact_candidates.begin(), exact_candidates.end()), exact_candidates.end());

        auto exact_deadline = start_time + chrono::milliseconds(1750);
        int exact_strategies = n > 5000 ? 1 : (n > 1500 ? 2 : (n > 700 ? 3 : 4));
        for (long long side : exact_candidates) {
            if (chrono::steady_clock::now() > exact_deadline) break;
            bool improved = false;
            for (int strategy = 0; strategy < exact_strategies; ++strategy) {
                if (chrono::steady_clock::now() > exact_deadline) break;
                vector<Placement> exact_place;
                if (exact_pack_side(pieces, side, exact_place, exact_deadline, strategy)) {
                    best_side = side;
                    best_place.swap(exact_place);
                    improved = true;
                    break;
                }
            }
            if (improved) break;
        }
    }

    cout << best_side << ' ' << best_side << '\n';
    for (int i = 0; i < n; ++i) {
        cout << best_place[i].x << ' ' << best_place[i].y << ' '
             << best_place[i].r << ' ' << best_place[i].f << '\n';
    }

    return 0;
}
// EVOLVE-BLOCK-END
