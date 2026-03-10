// Copyright 2024 KVCache.AI

#ifndef XGMI_TRANSPORT_H_
#define XGMI_TRANSPORT_H_

#include <string>

#include "cuda_alike.h"

#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "topology.h"
#include "transfer_metadata.h"
#include "transport/transport.h"

namespace mooncake {

class TransferMetadata;

struct XgmiPairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t h1 = std::hash<T1>{}(p.first);
        std::size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class XgmiTransport : public Transport {
   public:
    XgmiTransport();

    ~XgmiTransport();

    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest>& entries) override;

    Status submitTransferTask(
        const std::vector<TransferTask*>& task_list) override;

    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus& status) override;

   protected:
    int install(std::string& local_server_name,
                std::shared_ptr<TransferMetadata> meta,
                std::shared_ptr<Topology> topo) override;

    int registerLocalMemory(void* addr, size_t length,
                            const std::string& location, bool remote_accessible,
                            bool update_metadata = true) override;

    int unregisterLocalMemory(void* addr, bool update_metadata = true) override;

    int registerLocalMemoryBatch(const std::vector<BufferEntry>& buffer_list,
                                 const std::string& location) override;

    int unregisterLocalMemoryBatch(
        const std::vector<void*>& addr_list) override;

    int relocateSharedMemoryAddress(uint64_t& dest_addr, uint64_t length,
                                    uint64_t target_id);

    const char* getName() const override { return "xgmi"; }

   private:
    std::atomic_bool running_;

    struct OpenedShmEntry {
        void* shm_addr;
        uint64_t length;
    };

    std::unordered_map<std::pair<uint64_t, uint64_t>, OpenedShmEntry,
                       XgmiPairHash>
        remap_entries_;
    RWSpinlock remap_lock_;

    std::mutex register_mutex_;
};

}  // namespace mooncake

#endif  // XGMI_TRANSPORT_H_
