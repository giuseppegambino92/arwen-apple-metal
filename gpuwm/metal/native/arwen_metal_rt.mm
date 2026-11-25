// arwen_metal_rt.mm -- native Metal runtime for gpuwm's Metal backend.
//
// * Buffers are MTLStorageModeShared (Apple unified memory): the CPU address
//   and the GPU address name the same bytes, so host<->device transfers are
//   synchronisation only, never copies.
// * Kernels address memory through 64-bit GPU addresses carried in their
//   argument struct (bindless), so every allocation lives in one
//   MTLResidencySet attached to the queue.
// * Dispatches are encoded into one serial compute encoder per command buffer;
//   a buffer-scope memory barrier separates consecutive dispatches (resources
//   are untracked, so ordering is explicit).  Command buffers are committed
//   every `batch` dispatches and on synchronize().
// * A buffer released by Python while GPU work may still reference it is kept
//   alive until the command buffers committed so far have completed.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <mach/mach_time.h>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

// host seconds in the time base of MTLCommandBuffer GPUStartTime/GPUEndTime
double host_seconds() {
    static mach_timebase_info_data_t tb = [] { mach_timebase_info_data_t t; mach_timebase_info(&t); return t; }();
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

struct Runtime;

struct Buffer {
    id<MTLBuffer> buf = nil;
    size_t nbytes = 0;
    Runtime* rt = nullptr;
    uint64_t cpu() const { return (uint64_t)(uintptr_t)[buf contents]; }
    uint64_t gpu() const { return (uint64_t)[buf gpuAddress]; }
    ~Buffer();
};

struct Pipeline {
    id<MTLComputePipelineState> pso = nil;
    std::string name;
    int max_threads() const { return (int)[pso maxTotalThreadsPerThreadgroup]; }
    int exec_width() const { return (int)[pso threadExecutionWidth]; }
    int static_tg_mem() const { return (int)[pso staticThreadgroupMemoryLength]; }
};

struct Library {
    id<MTLLibrary> lib = nil;
    id<MTLDevice> dev = nil;
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        for (NSString* n in [lib functionNames]) out.push_back([n UTF8String]);
        return out;
    }
    Pipeline* pipeline(const std::string& name) {
        @autoreleasepool {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
            if (!fn) throw std::runtime_error("metal: no function " + name + " in library");
            MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
            d.computeFunction = fn;
            NSError* err = nil;
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone
                                                                              reflection:nil error:&err];
            if (!pso) throw std::runtime_error("metal: pipeline " + name + ": " + std::string([[err localizedDescription] UTF8String]));
            auto* p = new Pipeline();
            p->pso = pso;
            p->name = name;
            return p;
        }
    }
};

struct Runtime {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLResidencySet> residency = nil;
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    id<MTLCommandBuffer> last_committed = nil;
    id<MTLBuffer> null_buf = nil;
    // device-side abort word (CUDA __trap / asm("trap;")): bound at index 3 of every dispatch
    id<MTLBuffer> status_buf = nil;
    id<MTLEvent> order_event = nil;          // serialises command buffers (resources are untracked)
    int order_mode = 0;                      // 0: MTLEvent (GPU-only), 1: MTLSharedEvent, 2: none (diagnostic, unsafe)
    uint64_t order_value = 0;
    // command buffers committed / completed (they complete in commit order:
    // the order event chains them), for per-event completion queries
    std::atomic<uint64_t> n_committed{0}, n_completed{0};
    // GPUEndTime of completed command buffer #k (1-based, commit order = completion
    // order on the one queue), for event timestamps that do not depend on when the
    // host looks (cupy.cuda.get_elapsed_time)
    static constexpr size_t END_RING = 8192;
    std::vector<double> end_times = std::vector<double>(END_RING, 0.0);
    bool flush_on_alloc = false;
    bool profile = false;                    // one command buffer per dispatch, GPU time per kernel
    std::vector<std::pair<id<MTLCommandBuffer>, std::string>> prof_pending;
    std::vector<std::pair<std::string, double>> prof_done;             // debugging: new command buffer after each allocation
    bool concurrent = false;                 // serial encoder (+ barrier); true = concurrent + barriers
    NSMutableArray* deferred = nil;          // buffers to keep alive until the current cb completes
    // buffers released while committed work may still use them: (command buffer, buffer)
    std::vector<std::pair<id<MTLCommandBuffer>, id<MTLBuffer>>> zombies;
    std::recursive_mutex mu;
    int batch = 256;
    // commit after only `eager` dispatches when the GPU has run dry (after a
    // host synchronization, the GPU would otherwise wait for a full batch)
    int eager = 16;
    // ARWEN_METAL_TIMELINE: GPU [start, end] of every command buffer and the
    // host intervals spent in synchronize(), one time base (host seconds)
    bool timeline = false;
    std::mutex tl_mu;
    // buffers whose last command buffer completed, handed over by completion
    // handlers; those must not take mu: a thread holding mu can be blocked in
    // commandBuffer creation until a completed buffer's handlers have returned
    std::mutex done_mu;
    std::vector<NSArray*> done_keep;
    std::vector<double> tl_gpu;      // start, end pairs
    std::vector<double> tl_sync;     // begin, end pairs
    uint64_t n_eager = 0;
    int in_cb = 0;
    bool residency_dirty = false;
    bool barrier_between = true;
    uint64_t n_dispatch = 0, n_cb = 0, n_sync = 0;
    std::atomic<int> error_flag{0};
    std::string error_msg;
    int64_t live_bytes = 0, peak_bytes = 0;
    // pool of retired buffers by size class: a buffer enters only once no
    // committed or encoding work can reference it, so a reuse is a fresh buffer
    // for the GPU and for the CPU alike; pooled buffers stay resident
    std::unordered_map<size_t, std::vector<id<MTLBuffer>>> pool;
    int64_t pooled_bytes = 0, pool_cap = (int64_t)4096 << 20;
    uint64_t n_alloc = 0, n_pool_hits = 0;
    // the byte count each pooled buffer was last allocated for: a request for
    // the same count gets that buffer back first (CuPy's pool reuses a freed
    // block of the requested size, most recent first), any other one in the
    // class only if none matches
    std::unordered_map<void*, size_t> last_request;
    bool trace_pool = getenv("ARWEN_METAL_TRACE_POOL") != nullptr;

    static size_t size_class(size_t n) {
        const size_t page = 16384;
        if (n <= ((size_t)4 << 20)) return (n + page - 1) / page * page;
        size_t p = 1;
        while (p < n) p <<= 1;
        size_t step = p / 8;
        return (n + step - 1) / step * step;
    }

    // a buffer no work can reference any more: pool it or drop it
    void retire(id<MTLBuffer> b) {
        size_t len = [b length];
        if (trace_pool) fprintf(stderr, "[pool] retire %#llx len %zu\n", (unsigned long long)[b gpuAddress], len);
        if (pooled_bytes + (int64_t)len <= pool_cap && len == size_class(len)) {
            pool[len].push_back(b);
            pooled_bytes += (int64_t)len;
        } else {
            last_request.erase((__bridge void*)b);
            [residency removeAllocation:b];
            residency_dirty = true;
        }
    }

    // CuPy's MemoryPool.free_all_blocks: retired buffers go back to the system
    int64_t pool_release() {
        std::lock_guard<std::recursive_mutex> g(mu);
        int64_t freed = pooled_bytes;
        if (!pool.empty()) {
            for (auto& kv : pool) for (id<MTLBuffer> x : kv.second) [residency removeAllocation:x];
            pool.clear();
            last_request.clear();
            pooled_bytes = 0;
            [residency commit];
        }
        return freed;
    }

    int64_t pool_blocks() {
        std::lock_guard<std::recursive_mutex> g(mu);
        int64_t n = 0;
        for (auto& kv : pool) n += (int64_t)kv.second.size();
        return n;
    }

    Runtime() {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) throw std::runtime_error("metal: no Metal device");
        queue = [dev newCommandQueueWithMaxCommandBufferCount:64];
        MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
        rd.label = @"gpuwm";
        rd.initialCapacity = 4096;
        NSError* err = nil;
        residency = [dev newResidencySetWithDescriptor:rd error:&err];
        if (!residency) throw std::runtime_error("metal: residency set: " + std::string([[err localizedDescription] UTF8String]));
        [queue addResidencySet:residency];
        order_event = [dev newEvent];
        null_buf = [dev newBufferWithLength:256 options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        [residency addAllocation:null_buf];
        status_buf = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        memset([status_buf contents], 0, 64);   // word 0: abort word; word 4: the 64-bit atomicCAS lock
        [residency addAllocation:status_buf];
        [residency commit];
        deferred = [NSMutableArray new];
    }

    std::string name() const { return std::string([[dev name] UTF8String]); }
    int64_t max_threadgroup_memory() const { return (int64_t)[dev maxThreadgroupMemoryLength]; }

    Buffer* alloc(size_t nbytes) {
        std::lock_guard<std::recursive_mutex> g(mu);
        reap();                      // buffers whose work has completed are reusable now
        size_t n = nbytes ? nbytes : 16;
        size_t cls = size_class(n);
        n_alloc++;
        id<MTLBuffer> b = nil;
        auto it = pool.find(cls);
        if (it != pool.end() && !it->second.empty()) {
            auto& v = it->second;
            // among buffers last allocated for exactly n bytes, the lowest GPU
            // address: CuPy carves blocks from arenas and merges adjacent free
            // ones, so after a workspace is released a request of the same size
            // starts again at the arena's beginning -- the earliest allocation
            size_t pick = v.size() - 1;
            uint64_t best = UINT64_MAX;
            for (size_t i = 0; i < v.size(); ++i) {
                auto lr = last_request.find((__bridge void*)v[i]);
                if (lr != last_request.end() && lr->second == n && (uint64_t)[v[i] gpuAddress] < best) {
                    best = (uint64_t)[v[i] gpuAddress];
                    pick = i;
                }
            }
            b = v[pick];
            v.erase(v.begin() + (ptrdiff_t)pick);
            pooled_bytes -= (int64_t)cls;
            n_pool_hits++;
        } else {
            b = [dev newBufferWithLength:cls options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
            if (!b && pooled_bytes) {                  // memory pressure: give the pool back and retry
                for (auto& kv : pool) for (id<MTLBuffer> x : kv.second) [residency removeAllocation:x];
                pool.clear();
                last_request.clear();
                pooled_bytes = 0;
                [residency commit];
                b = [dev newBufferWithLength:cls options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
            }
            if (!b) throw std::runtime_error("metal: out of memory allocating " + std::to_string(n) + " bytes");
            [residency addAllocation:b];
            residency_dirty = true;
        }
        // the residency set is committed when the next command buffer begins; work
        // already being encoded must see this allocation resident too
        if (enc != nil) {
            if (flush_on_alloc) commit();
            else [enc useResource:b usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
        }
        last_request[(__bridge void*)b] = n;
        if (trace_pool) fprintf(stderr, "[pool] alloc %zu -> %#llx\n", n, (unsigned long long)[b gpuAddress]);
        auto* r = new Buffer();
        r->buf = b;
        r->nbytes = n;
        r->rt = this;
        live_bytes += (int64_t)n;
        if (live_bytes > peak_bytes) peak_bytes = live_bytes;
        return r;
    }

    void release(id<MTLBuffer> b, size_t n) {
        std::lock_guard<std::recursive_mutex> g(mu);
        if (trace_pool) fprintf(stderr, "[pool] release %#llx n %zu cb %d\n", (unsigned long long)[b gpuAddress], n, cb != nil);
        live_bytes -= (int64_t)n;
        if (cb != nil) {
            // referenced by work still being encoded: released when that command buffer completes
            [deferred addObject:b];
        } else if (last_committed != nil && [last_committed status] < MTLCommandBufferStatusCompleted) {
            // referenced by committed, in-flight work: reaped once it has completed
            zombies.emplace_back(last_committed, b);
        } else {
            retire(b);
        }
    }

    void reap() {
        {
            std::vector<NSArray*> ready;
            {
                std::lock_guard<std::mutex> dg(done_mu);
                ready.swap(done_keep);
            }
            for (NSArray* keep : ready)
                for (id<MTLBuffer> b in keep) retire(b);
        }
        size_t k = 0;
        for (size_t i = 0; i < zombies.size(); ++i) {
            if ([zombies[i].first status] >= MTLCommandBufferStatusCompleted) {
                retire(zombies[i].second);
            } else {
                zombies[k++] = zombies[i];
            }
        }
        zombies.resize(k);
    }

    void begin() {
        if (cb == nil) {
            reap();
            if (residency_dirty) { [residency commit]; residency_dirty = false; }
            cb = [queue commandBufferWithUnretainedReferences];
            Runtime* self_ = this;
            [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
                uint64_t k = self_->n_completed.load() + 1;
                self_->end_times[k % Runtime::END_RING] = [c GPUEndTime];
                self_->n_completed.fetch_add(1);
                if (self_->timeline) {
                    std::lock_guard<std::mutex> tg(self_->tl_mu);
                    self_->tl_gpu.push_back([c GPUStartTime]);
                    self_->tl_gpu.push_back([c GPUEndTime]);
                }
                if ([c status] == MTLCommandBufferStatusError && self_->error_flag.exchange(1) == 0) {
                    NSError* e = [c error];
                    self_->error_msg = e ? std::string([[e description] UTF8String]) : "unknown command buffer error";
                }
            }];
            // wait for the previous command buffer: with untracked resources the
            // GPU would otherwise overlap consecutive command buffers
            if (order_value && order_mode != 2) [cb encodeWaitForEvent:order_event value:order_value];
            enc = [cb computeCommandEncoderWithDispatchType:(concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial)];
            in_cb = 0;
            n_cb++;
        }
    }

    void commit() {
        if (cb == nil) return;
        [enc endEncoding];
        if (order_mode != 2) [cb encodeSignalEvent:order_event value:++order_value];
        // buffers released while this command buffer was being encoded: reaped
        // once its status reads completed (a completion handler runs later than
        // waitUntilCompleted returns, so a pool fed by the handler missed buffers
        // the host had already synchronised on -- CuPy reuses them at once)
        for (id<MTLBuffer> b in deferred) zombies.emplace_back(cb, b);
        [deferred removeAllObjects];
        [cb commit];
        n_committed.fetch_add(1);
        last_committed = cb;
        cb = nil;
        enc = nil;
    }

    void dispatch(Pipeline* p, nb::bytes args, nb::bytes launch, Buffer* globals,
                  uint32_t gx, uint32_t gy, uint32_t gz, uint32_t bx, uint32_t by, uint32_t bz, uint32_t tg_mem) {
        std::lock_guard<std::recursive_mutex> g(mu);
        check_error();
        if (gx == 0 || gy == 0 || gz == 0) return;
        begin();
        if (in_cb > 0 && barrier_between) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:p->pso];
        size_t na = args.size();
        static const uint64_t zero[2] = {0, 0};
        [enc setBytes:(na ? args.c_str() : (const char*)zero) length:(na ? na : 16) atIndex:0];
        [enc setBytes:launch.c_str() length:launch.size() atIndex:1];
        [enc setBuffer:(globals ? globals->buf : null_buf) offset:0 atIndex:2];
        [enc setBuffer:status_buf offset:0 atIndex:3];
        if (tg_mem) [enc setThreadgroupMemoryLength:((tg_mem + 15) / 16) * 16 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(bx, by, bz)];
        n_dispatch++;
        if (profile) {
            id<MTLCommandBuffer> mine = cb;
            commit();
            prof_pending.emplace_back(mine, p->name);
            return;
        }
        ++in_cb;
        if (in_cb >= batch) {
            commit();
        } else if (eager > 0 && in_cb >= eager &&
                   (last_committed == nil || [last_committed status] >= MTLCommandBufferStatusCompleted)) {
            n_eager++;
            commit();
        }
    }

    void flush() {
        std::lock_guard<std::recursive_mutex> g(mu);
        commit();
    }

    void synchronize() {
        double t0 = timeline ? host_seconds() : 0.0;
        id<MTLCommandBuffer> wait = nil;
        {
            std::lock_guard<std::recursive_mutex> g(mu);
            commit();
            wait = last_committed;
            n_sync++;
        }
        if (wait != nil) [wait waitUntilCompleted];
        {
            std::lock_guard<std::recursive_mutex> g(mu);
            reap();
        }
        if (timeline) {
            std::lock_guard<std::mutex> tg(tl_mu);
            tl_sync.push_back(t0);
            tl_sync.push_back(host_seconds());
        }
        check_error();
    }

    bool busy() {
        std::lock_guard<std::recursive_mutex> g(mu);
        return cb != nil || (last_committed != nil && [last_committed status] < MTLCommandBufferStatusCompleted);
    }

    void check_error() {
        if (error_flag.load()) {
            error_flag.store(0);
            throw std::runtime_error("metal: GPU command buffer failed: " + error_msg);
        }
    }

    Library* load_library(const std::string& path) {
        @autoreleasepool {
            NSError* err = nil;
            NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
            id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
            if (!lib) throw std::runtime_error("metal: cannot load " + path + ": " + std::string([[err localizedDescription] UTF8String]));
            auto* l = new Library();
            l->lib = lib;
            l->dev = dev;
            return l;
        }
    }

    std::vector<std::pair<std::string, double>> profile_drain() {
        std::lock_guard<std::recursive_mutex> g(mu);
        size_t k = 0;
        for (size_t i = 0; i < prof_pending.size(); ++i) {
            id<MTLCommandBuffer> c = prof_pending[i].first;
            if ([c status] >= MTLCommandBufferStatusCompleted) {
                prof_done.emplace_back(prof_pending[i].second, [c GPUEndTime] - [c GPUStartTime]);
            } else {
                prof_pending[k++] = prof_pending[i];
            }
        }
        prof_pending.resize(k);
        std::vector<std::pair<std::string, double>> out;
        out.swap(prof_done);
        return out;
    }

    nb::tuple timeline_drain() {
        std::lock_guard<std::mutex> tg(tl_mu);
        std::vector<double> g, s;
        g.swap(tl_gpu);
        s.swap(tl_sync);
        return nb::make_tuple(g, s, host_seconds());
    }

    // the abort word, cleared on read (call after synchronize)
    uint32_t status_take() {
        std::lock_guard<std::recursive_mutex> g(mu);
        volatile uint32_t* w = (volatile uint32_t*)[status_buf contents];
        uint32_t v = w[0];
        w[0] = 0;
        return v;
    }

    nb::dict stats() {
        nb::dict d;
        d["dispatches"] = n_dispatch;
        d["command_buffers"] = n_cb;
        d["synchronizations"] = n_sync;
        d["eager_commits"] = n_eager;
        d["allocations"] = n_alloc;
        d["pool_hits"] = n_pool_hits;
        d["pooled_bytes"] = pooled_bytes;
        d["live_bytes"] = live_bytes;
        d["peak_bytes"] = peak_bytes;
        d["recommended_max_working_set"] = (uint64_t)[dev recommendedMaxWorkingSetSize];
        d["current_allocated_size"] = (uint64_t)[dev currentAllocatedSize];
        return d;
    }
};

Buffer::~Buffer() {
    if (buf != nil && rt != nullptr) rt->release(buf, nbytes);
    buf = nil;
}

}  // namespace

NB_MODULE(_arwen_metal_rt, m) {
    nb::class_<Buffer>(m, "Buffer")
        .def_prop_ro("cpu_address", &Buffer::cpu)
        .def_prop_ro("gpu_address", &Buffer::gpu)
        .def_ro("nbytes", &Buffer::nbytes);
    nb::class_<Pipeline>(m, "Pipeline")
        .def_prop_ro("max_threads", &Pipeline::max_threads)
        .def_prop_ro("exec_width", &Pipeline::exec_width)
        .def_prop_ro("static_threadgroup_memory", &Pipeline::static_tg_mem)
        .def_ro("name", &Pipeline::name);
    nb::class_<Library>(m, "Library")
        .def("function_names", &Library::names)
        .def("pipeline", &Library::pipeline, nb::rv_policy::take_ownership);
    nb::class_<Runtime>(m, "Runtime")
        .def(nb::init<>())
        .def_prop_ro("name", &Runtime::name)
        .def_rw("batch", &Runtime::batch)
        .def_rw("eager", &Runtime::eager)
        .def_rw("pool_cap", &Runtime::pool_cap)
        .def("pool_release", &Runtime::pool_release)
        .def("max_threadgroup_memory", &Runtime::max_threadgroup_memory)
        .def("pool_blocks", &Runtime::pool_blocks)
        .def_rw("timeline", &Runtime::timeline)
        .def("set_order_mode", [](Runtime& r, int m) {
            r.order_mode = m;
            r.order_event = (m == 1) ? (id<MTLEvent>)[r.dev newSharedEvent] : [r.dev newEvent];
            r.order_value = 0;
        })
        .def("timeline_drain", &Runtime::timeline_drain)
        .def("status_take", &Runtime::status_take)
        .def_rw("barrier_between", &Runtime::barrier_between)
        .def_rw("concurrent", &Runtime::concurrent)
        .def_rw("flush_on_alloc", &Runtime::flush_on_alloc)
        .def_rw("profile", &Runtime::profile)
        .def("profile_drain", &Runtime::profile_drain)
        .def("alloc", &Runtime::alloc, nb::rv_policy::take_ownership, nb::keep_alive<0, 1>())
        .def("load_library", &Runtime::load_library, nb::rv_policy::take_ownership)
        .def("dispatch", &Runtime::dispatch, nb::arg("pipeline"), nb::arg("args"), nb::arg("launch"),
             nb::arg("globals").none(), nb::arg("gx"), nb::arg("gy"), nb::arg("gz"), nb::arg("bx"), nb::arg("by"),
             nb::arg("bz"), nb::arg("tg_mem"))
        .def("flush", &Runtime::flush)
        .def("synchronize", &Runtime::synchronize, nb::call_guard<nb::gil_scoped_release>())
        .def("busy", &Runtime::busy)
        .def("committed", [](Runtime& r) { return (uint64_t)r.n_committed.load(); })
        .def("completed", [](Runtime& r) { return (uint64_t)r.n_completed.load(); })
        .def("end_time", [](Runtime& r, uint64_t k) {
            // seconds, in host_seconds' time base; NaN when #k has not completed or left the ring
            uint64_t done = r.n_completed.load();
            if (k == 0 || k > done || done - k >= Runtime::END_RING) return std::nan("");
            return r.end_times[k % Runtime::END_RING]; })
        .def("host_seconds", [](Runtime&) { return host_seconds(); })
        .def("stats", &Runtime::stats);
}
