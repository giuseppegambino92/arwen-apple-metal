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
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

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
    id<MTLSharedEvent> order_event = nil;    // serialises command buffers (resources are untracked)
    uint64_t order_value = 0;
    bool flush_on_alloc = false;             // debugging: new command buffer after each allocation
    bool concurrent = false;                 // serial encoder (+ barrier); true = concurrent + barriers
    NSMutableArray* deferred = nil;          // buffers to keep alive until the current cb completes
    // buffers released while committed work may still use them: (command buffer, buffer)
    std::vector<std::pair<id<MTLCommandBuffer>, id<MTLBuffer>>> zombies;
    std::recursive_mutex mu;
    int batch = 256;
    int in_cb = 0;
    bool residency_dirty = false;
    bool barrier_between = true;
    uint64_t n_dispatch = 0, n_cb = 0, n_sync = 0;
    std::atomic<int> error_flag{0};
    std::string error_msg;
    int64_t live_bytes = 0, peak_bytes = 0;

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
        order_event = [dev newSharedEvent];
        null_buf = [dev newBufferWithLength:256 options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        [residency addAllocation:null_buf];
        [residency commit];
        deferred = [NSMutableArray new];
    }

    std::string name() const { return std::string([[dev name] UTF8String]); }

    Buffer* alloc(size_t nbytes) {
        std::lock_guard<std::recursive_mutex> g(mu);
        size_t n = nbytes ? nbytes : 16;
        id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        if (!b) throw std::runtime_error("metal: out of memory allocating " + std::to_string(n) + " bytes");
        [residency addAllocation:b];
        residency_dirty = true;
        // the residency set is committed when the next command buffer begins; work
        // already being encoded must see this allocation resident too
        if (enc != nil) {
            if (flush_on_alloc) commit();
            else [enc useResource:b usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
        }
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
        live_bytes -= (int64_t)n;
        if (cb != nil) {
            // referenced by work still being encoded: released when that command buffer completes
            [deferred addObject:b];
        } else if (last_committed != nil && [last_committed status] < MTLCommandBufferStatusCompleted) {
            // referenced by committed, in-flight work: reaped once it has completed
            zombies.emplace_back(last_committed, b);
        } else {
            [residency removeAllocation:b];
            residency_dirty = true;
        }
    }

    void reap() {
        size_t k = 0;
        for (size_t i = 0; i < zombies.size(); ++i) {
            if ([zombies[i].first status] >= MTLCommandBufferStatusCompleted) {
                [residency removeAllocation:zombies[i].second];
                residency_dirty = true;
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
                if ([c status] == MTLCommandBufferStatusError && self_->error_flag.exchange(1) == 0) {
                    NSError* e = [c error];
                    self_->error_msg = e ? std::string([[e description] UTF8String]) : "unknown command buffer error";
                }
            }];
            // wait for the previous command buffer: with untracked resources the
            // GPU would otherwise overlap consecutive command buffers
            if (order_value) [cb encodeWaitForEvent:order_event value:order_value];
            enc = [cb computeCommandEncoderWithDispatchType:(concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial)];
            in_cb = 0;
            n_cb++;
        }
    }

    void commit() {
        if (cb == nil) return;
        [enc endEncoding];
        [cb encodeSignalEvent:order_event value:++order_value];
        if ([deferred count]) {
            NSArray* keep = [deferred copy];
            [deferred removeAllObjects];
            id<MTLResidencySet> rs = residency;
            Runtime* self_ = this;
            [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
                std::lock_guard<std::recursive_mutex> g(self_->mu);
                for (id<MTLBuffer> b in keep) [rs removeAllocation:b];
                self_->residency_dirty = true;
            }];
        }
        [cb commit];
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
        if (tg_mem) [enc setThreadgroupMemoryLength:((tg_mem + 15) / 16) * 16 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(bx, by, bz)];
        n_dispatch++;
        if (++in_cb >= batch) commit();
    }

    void flush() {
        std::lock_guard<std::recursive_mutex> g(mu);
        commit();
    }

    void synchronize() {
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

    nb::dict stats() {
        nb::dict d;
        d["dispatches"] = n_dispatch;
        d["command_buffers"] = n_cb;
        d["synchronizations"] = n_sync;
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
        .def_rw("barrier_between", &Runtime::barrier_between)
        .def_rw("concurrent", &Runtime::concurrent)
        .def_rw("flush_on_alloc", &Runtime::flush_on_alloc)
        .def("alloc", &Runtime::alloc, nb::rv_policy::take_ownership, nb::keep_alive<0, 1>())
        .def("load_library", &Runtime::load_library, nb::rv_policy::take_ownership)
        .def("dispatch", &Runtime::dispatch, nb::arg("pipeline"), nb::arg("args"), nb::arg("launch"),
             nb::arg("globals").none(), nb::arg("gx"), nb::arg("gy"), nb::arg("gz"), nb::arg("bx"), nb::arg("by"),
             nb::arg("bz"), nb::arg("tg_mem"))
        .def("flush", &Runtime::flush)
        .def("synchronize", &Runtime::synchronize, nb::call_guard<nb::gil_scoped_release>())
        .def("busy", &Runtime::busy)
        .def("stats", &Runtime::stats);
}
