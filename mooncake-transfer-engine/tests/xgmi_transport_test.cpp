#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <thread>
#include <memory>
#include <cstring>

#include "cuda_alike.h"
#include "transfer_engine.h"
#include "transport/transport.h"

using namespace mooncake;

static void checkGpuError(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        LOG(ERROR) << message << " (Error code: " << result << " - "
                   << cudaGetErrorString(result) << ")";
        exit(EXIT_FAILURE);
    }
}

static void* allocateGpuBuffer(size_t size, int gpu_id) {
    checkGpuError(cudaSetDevice(gpu_id), "Failed to set device");
    void* d_buf = nullptr;
    checkGpuError(cudaMalloc(&d_buf, size),
                  "Failed to allocate device memory");
    return d_buf;
}

static void freeGpuBuffer(void* addr) {
    checkGpuError(cudaFree(addr), "Failed to free device memory");
}

class XgmiTransportTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("XgmiTransportTest");
        FLAGS_logtostderr = 1;

        const char* env = std::getenv("MC_METADATA_SERVER");
        metadata_server = env ? env : "P2PHANDSHAKE";

        env = std::getenv("MC_LOCAL_SERVER_NAME");
        local_server_name = env ? env : "127.0.0.1:12345";

        env = std::getenv("MC_GPU_ID");
        gpu_id = env ? std::atoi(env) : 0;
    }

    void TearDown() override {
        google::ShutdownGoogleLogging();
    }

    std::string metadata_server;
    std::string local_server_name;
    int gpu_id;
};

TEST_F(XgmiTransportTest, WriteAndRead) {
    const size_t kDataLength = 4096000;

    // Single-engine loopback test
    auto engine = std::make_unique<TransferEngine>(false);
    auto hostname_port = parseHostNameWithPort(local_server_name);
    auto rc = engine->init(metadata_server, local_server_name,
                           hostname_port.first.c_str(), hostname_port.second);
    ASSERT_EQ(rc, 0);

    Transport* xport = engine->installTransport("xgmi", nullptr);
    ASSERT_NE(xport, nullptr);

    // Allocate GPU buffer: first half for write source, second half for read dest
    void* gpu_buffer = allocateGpuBuffer(kDataLength * 2, gpu_id);
    rc = engine->registerLocalMemory(gpu_buffer, kDataLength * 2,
                                     "hip:" + std::to_string(gpu_id));
    ASSERT_EQ(rc, 0);

    // Fill first half with known data
    std::vector<char> host_data(kDataLength, 'A');
    checkGpuError(cudaMemcpy(gpu_buffer, host_data.data(), kDataLength,
                             cudaMemcpyHostToDevice),
                  "Memcpy H2D");

    // Zero out second half
    checkGpuError(cudaMemset((char*)gpu_buffer + kDataLength, 0, kDataLength),
                  "Memset second half");

    // In P2PHANDSHAKE mode, the engine listens on a dynamic port
    auto actual_server_name = engine->getLocalIpAndPort();
    LOG(INFO) << "XgmiTransportTest: actual server address: " << actual_server_name;
    auto segment_id = engine->openSegment(actual_server_name);
    auto segment_desc = engine->getMetadata()->getSegmentDescByID(segment_id);
    ASSERT_NE(segment_desc, nullptr);
    uint64_t remote_base = segment_desc->buffers[0].addr;

    // Write: first half -> second half (via xGMI transport loopback)
    {
        auto batch_id = engine->allocateBatchID(1);
        TransferRequest entry;
        entry.opcode = TransferRequest::WRITE;
        entry.length = kDataLength;
        entry.source = gpu_buffer;
        entry.target_id = segment_id;
        entry.target_offset = remote_base + kDataLength;
        Status s = engine->submitTransfer(batch_id, {entry});
        ASSERT_TRUE(s.ok());

        TransferStatus status;
        do {
            s = engine->getTransferStatus(batch_id, 0, status);
            ASSERT_TRUE(s.ok());
        } while (status.s == TransferStatusEnum::WAITING);

        ASSERT_EQ(status.s, TransferStatusEnum::COMPLETED);
        s = engine->freeBatchID(batch_id);
        ASSERT_TRUE(s.ok());
    }

    // Verify: read second half back to host and compare
    std::vector<char> host_check(kDataLength);
    checkGpuError(
        cudaMemcpy(host_check.data(), (char*)gpu_buffer + kDataLength,
                   kDataLength, cudaMemcpyDeviceToHost),
        "Memcpy D2H check");

    for (size_t i = 0; i < kDataLength; ++i) {
        ASSERT_EQ(host_check[i], 'A') << "Mismatch at byte " << i;
    }

    // Read: second half -> first half (via xGMI transport loopback READ)
    // First clear the first half
    checkGpuError(cudaMemset(gpu_buffer, 0, kDataLength),
                  "Memset first half");

    {
        auto batch_id = engine->allocateBatchID(1);
        TransferRequest entry;
        entry.opcode = TransferRequest::READ;
        entry.length = kDataLength;
        entry.source = gpu_buffer;
        entry.target_id = segment_id;
        entry.target_offset = remote_base + kDataLength;
        Status s = engine->submitTransfer(batch_id, {entry});
        ASSERT_TRUE(s.ok());

        TransferStatus status;
        do {
            s = engine->getTransferStatus(batch_id, 0, status);
            ASSERT_TRUE(s.ok());
        } while (status.s == TransferStatusEnum::WAITING);

        ASSERT_EQ(status.s, TransferStatusEnum::COMPLETED);
        s = engine->freeBatchID(batch_id);
        ASSERT_TRUE(s.ok());
    }

    // Verify read result
    checkGpuError(
        cudaMemcpy(host_check.data(), gpu_buffer,
                   kDataLength, cudaMemcpyDeviceToHost),
        "Memcpy D2H read check");

    for (size_t i = 0; i < kDataLength; ++i) {
        ASSERT_EQ(host_check[i], 'A') << "Read mismatch at byte " << i;
    }

    // Cleanup
    engine->unregisterLocalMemory(gpu_buffer);
    freeGpuBuffer(gpu_buffer);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
