# dma — Reserved for v2 (FPGA)

This block does not exist in v1 (D8): the host writes directly into the buffers.

To be specified in v2:
- master interface to the external memory (Wishbone or AXI);
- source and destination address registers;
- valid/ready handshake at the boundaries (D12), because the latency of the external memory is variable;
- possibly double buffering (D6), to overlap the transfers with the computation.
