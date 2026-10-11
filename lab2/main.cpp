#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <execution>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace std;

using Vec = vector<double>;

volatile double g_sink = 0;

template <class F>
double measureUs(F&& f, int repeats)
{
    vector<double> times;
    times.reserve(repeats);
    for (int i = 0; i < repeats; ++i)
    {
        auto start = chrono::steady_clock::now();
        f();
        auto stop = chrono::steady_clock::now();
        times.push_back(chrono::duration<double, micro>(stop - start).count());
    }
    sort(times.begin(), times.end());
    return times[times.size() / 2];
}

class SequenceStore
{
public:
    static string fileName(size_t n) { return "data_" + to_string(n) + ".txt"; }

    static void ensureExists(size_t n)
    {
        if (ifstream(fileName(n)).good())
            return;
        mt19937_64 gen(12345 + n);
        uniform_real_distribution<double> dist(1.0, 100.0);
        ofstream out(fileName(n));
        out << n << '\n';
        out.precision(6);
        for (size_t i = 0; i < n; ++i)
            out << dist(gen) << (i % 10 == 9 ? '\n' : ' ');
    }

    static Vec load(size_t n)
    {
        ensureExists(n);
        ifstream in(fileName(n));
        size_t count = 0;
        in >> count;
        Vec values(count);
        for (auto& x : values)
            in >> x;
        return values;
    }
};

struct Dataset
{
    size_t n;
    Vec values;
};

struct FastOp
{
    static constexpr const char* name = "fast (2x+1)";
    double operator()(double x) const { return 2.0 * x + 1.0; }
};

struct SlowOp
{
    static constexpr const char* name = "slow (sin/cos/sqrt/log/exp)";
    double operator()(double x) const
    {
        return sin(x) * cos(x) + sqrt(x) + log(x + 1.0) + exp(-x / 50.0);
    }
};

template <class InIt, class OutIt, class F>
OutIt parallelTransform(InIt first, InIt last, OutIt result, F f, unsigned k)
{
    using Value = remove_cvref_t<decltype(f(*first))>;
    const size_t n = distance(first, last);
    k = static_cast<unsigned>(max<size_t>(1, min<size_t>(k, n)));

    vector<vector<Value>> parts(k);
    {
        vector<jthread> threads;
        threads.reserve(k);
        const size_t base = n / k, extra = n % k;
        size_t offset = 0;
        for (unsigned i = 0; i < k; ++i)
        {
            const size_t len = base + (i < extra ? 1 : 0);
            InIt b = first + offset, e = b + len;
            threads.emplace_back([&parts, i, b, e, len, f]
                {
                    parts[i].resize(len);
                    transform(b, e, parts[i].begin(), f); });
            offset += len;
        }
    }

    for (auto& part : parts)
        result = copy(part.begin(), part.end(), result);
    return result;
}

void printTableHeader(const char* corner, const vector<Dataset>& sets)
{
    printf("%-12s", corner);
    for (const auto& s : sets)
        printf("%14zu", s.n);
    printf("\n");
}

template <class Op>
void studyLibrary(const Op& op, const vector<Dataset>& sets, int repeats)
{
    using Runner = function<void(const Vec&, Vec&)>;
    const vector<pair<const char*, Runner>> runners = {
        {"no policy", [&](const Vec& in, Vec& out)
         { transform(in.begin(), in.end(), out.begin(), op); }},
        {"seq", [&](const Vec& in, Vec& out)
         { transform(execution::seq, in.begin(), in.end(), out.begin(), op); }},
        {"par", [&](const Vec& in, Vec& out)
         { transform(execution::par, in.begin(), in.end(), out.begin(), op); }},
        {"unseq", [&](const Vec& in, Vec& out)
         { transform(execution::unseq, in.begin(), in.end(), out.begin(), op); }},
        {"par_unseq", [&](const Vec& in, Vec& out)
         { transform(execution::par_unseq, in.begin(), in.end(), out.begin(), op); }},
    };

    printf("\n[Library transform] operation: %s; time in us (median of %d); columns = N\n",
        Op::name, repeats);
    printTableHeader("policy", sets);
    for (const auto& [name, run] : runners)
    {
        printf("%-12s", name);
        for (const auto& s : sets)
        {
            Vec out(s.n);
            printf("%14.1f", measureUs([&] { run(s.values, out); }, repeats));
            g_sink = out.back();
        }
        printf("\n");
    }
}

vector<unsigned> makeKValues(unsigned hw)
{
    vector<unsigned> ks;
    for (unsigned k = 1; k <= 2 * hw; ++k)
        ks.push_back(k);
    for (unsigned mult : {4u, 8u, 16u, 32u})
        ks.push_back(mult * hw);
    return ks;
}

struct CustomResult
{
    vector<unsigned> ks;
    vector<vector<double>> timesUs;
    bool allCorrect = true;
};

template <class Op>
CustomResult measureCustom(const Op& op, const vector<Dataset>& sets, int repeats, unsigned hw)
{
    CustomResult res;
    res.ks = makeKValues(hw);
    res.timesUs.assign(res.ks.size(), vector<double>(sets.size()));

    for (size_t j = 0; j < sets.size(); ++j)
    {
        const Dataset& s = sets[j];
        Vec expected(s.n), out(s.n);
        transform(s.values.begin(), s.values.end(), expected.begin(), op);
        for (size_t i = 0; i < res.ks.size(); ++i)
        {
            res.timesUs[i][j] = measureUs([&]
                { parallelTransform(s.values.begin(), s.values.end(), out.begin(), op, res.ks[i]); },
                repeats);
            res.allCorrect = res.allCorrect && (out == expected);
            g_sink = out.back();
        }
    }
    return res;
}

void printCustomTable(const CustomResult& res, const vector<Dataset>& sets)
{
    printTableHeader("K", sets);
    for (size_t i = 0; i < res.ks.size(); ++i)
    {
        printf("%-12u", res.ks[i]);
        for (double t : res.timesUs[i])
            printf("%14.1f", t);
        printf("\n");
    }
}

void printBestK(const CustomResult& res, const vector<Dataset>& sets, unsigned hw)
{
    printf("Result equals std::transform for every K: %s\n", res.allCorrect ? "yes" : "NO (ERROR)");
    printf("Hardware (logical) threads: %u\n", hw);
    for (size_t j = 0; j < sets.size(); ++j)
    {
        size_t best = 0;
        for (size_t i = 1; i < res.ks.size(); ++i)
            if (res.timesUs[i][j] < res.timesUs[best][j])
                best = i;
        printf("N=%-9zu best K=%-4u (%.1f us), K / threads = %.2f\n",
            sets[j].n, res.ks[best], res.timesUs[best][j], double(res.ks[best]) / hw);
    }
}

template <class Op>
void studyCustom(const Op& op, const vector<Dataset>& sets, int repeats)
{
    const unsigned hw = max(1u, thread::hardware_concurrency());
    const CustomResult res = measureCustom(op, sets, repeats, hw);

    printf("\n[Custom parallel transform] operation: %s; time in us (median of %d); columns = N\n",
        Op::name, repeats);
    printCustomTable(res, sets);
    printBestK(res, sets, hw);
}

int main()
{
#if defined(NO_OPT)
    printf("Build mode: not optimized\n");
#elif defined(__OPTIMIZE__) || (defined(_MSC_VER) && !defined(_DEBUG))
    printf("Build mode: optimized\n");
#else
    printf("Build mode: not optimized\n");
#endif
    const vector<size_t> sizes = { 1000, 100000, 1000000, 5000000 };
    const int repeats = 5;

    vector<Dataset> sets;
    for (auto n : sizes)
        sets.push_back({ n, SequenceStore::load(n) });

    studyLibrary(FastOp{}, sets, repeats);
    studyLibrary(SlowOp{}, sets, repeats);
    studyCustom(FastOp{}, sets, repeats);
    studyCustom(SlowOp{}, sets, repeats);
}