"""
npu_top — NPU assembly (specification: src/npu_top/interface.md)

The only block visible from the outside: a 32-bit parallel bus and an interrupt line.
It instantiates and wires all the other blocks, with no logic of its own:

  host_interface ──► controller, post_processing (configuration), memories (host ports)
  controller ──► weight_buffer / activation_buffer (reads), data_feeder, output_drain
  weight_buffer, activation_buffer ──► data_feeder ──► pe_array ──► output_drain ──► post_processing
  post_processing ──► activation_buffer (write of Y)

DATAFLOW ('OS' or 'WS') selects the variant of the compute core. v1 requires N = 4 (32-bit word on the bus).
"""

from py4hw import Logic, Wire

from src.host_interface.host_interface import HostInterface
from src.controller.controller import Controller_OS, Controller_WS
from src.weight_buffer.weight_buffer import WeightBuffer
from src.activation_buffer.activation_buffer import ActivationBuffer
from src.data_feeder.data_feeder import DataFeeder_OS, DataFeeder_WS
from src.pe_array.pe_array import PE_Array_OS, PE_Array_WS
from src.output_drain.output_drain import OutputDrain_OS, OutputDrain_WS
from src.post_processing.post_processing import PostProcessing_OS, PostProcessing_WS


class NpuTop(Logic):

    def __init__(self, parent, name: str, rst_n: Wire,
                 bus_addr: Wire, bus_wdata: Wire, bus_we: Wire, bus_re: Wire, bus_rdata: Wire, irq: Wire,
                 dataflow: str = 'OS', DIM_MAX: int = 64):
        super().__init__(parent, name)

        rst_n = self.addIn('rst_n', rst_n)
        bus_addr = self.addIn('bus_addr', bus_addr)
        bus_wdata = self.addIn('bus_wdata', bus_wdata)
        bus_we = self.addIn('bus_we', bus_we)
        bus_re = self.addIn('bus_re', bus_re)
        bus_rdata = self.addOut('bus_rdata', bus_rdata)
        irq = self.addOut('irq', irq)

        assert dataflow in ('OS', 'WS')
        self.dataflow = dataflow
        N, DW, AW = 4, 8, 32
        IW = DIM_MAX.bit_length() - 1                     # log2(DIM_MAX)
        TW = (DIM_MAX // N).bit_length() - 1              # log2(DIM_MAX / N)
        BAW = TW + IW
        TAG_W = BAW + N + IW
        w = lambda name, width=1: self.wire(name, width)

        # configuration and status
        dim_m, dim_k, dim_n = w('dim_m', IW + 1), w('dim_k', IW + 1), w('dim_n', IW + 1)
        req_mult, req_shift, relu_en = w('req_mult', 16), w('req_shift', 5), w('relu_en')
        go, busy, done, cycles = w('go'), w('busy'), w('done'), w('cycles', 32)
        # host ports of the memories
        host_wdata = w('host_wdata', 32)
        wb_addr, wb_we, wb_re, wb_rdata = w('wb_addr', BAW), w('wb_we'), w('wb_re'), w('wb_rdata', 32)
        ab_addr, ab_we, ab_re, ab_rdata = w('ab_addr', BAW + 1), w('ab_we'), w('ab_re'), w('ab_rdata', 32)
        bias_addr, bias_we, bias_re, bias_rdata = w('bias_addr', IW), w('bias_we'), w('bias_re'), w('bias_rdata', AW)
        # controller commands
        w_addr, w_re, x_addr, x_re = w('w_addr', BAW), w('w_re'), w('x_addr', BAW), w('x_re')
        w_mask, x_mask, ctl, tag = w('w_mask', N), w('x_mask', N), w('ctl'), w('tag', TAG_W)
        # datapath
        w_rdata, x_rdata = w('w_rdata', N * DW), w('x_rdata', N * DW)
        w_arr, x_arr, ctl_arr = w('w_arr', N * DW), w('x_arr', N * DW), w('ctl_arr')
        res, res_valid, tag_out = w('res', N * AW), w('res_valid'), w('tag_out', TAG_W)
        y_addr, y_wdata, y_we, y_wmask = w('y_addr', BAW), w('y_wdata', N * DW), w('y_we'), w('y_wmask', N)

        self.host = HostInterface(self, 'host_interface', rst_n, bus_addr, bus_wdata, bus_we, bus_re, bus_rdata,
                                  dim_m, dim_k, dim_n, req_mult, req_shift, relu_en, go, busy, done, cycles,
                                  host_wdata, wb_addr, wb_we, wb_re, wb_rdata, ab_addr, ab_we, ab_re, ab_rdata,
                                  bias_addr, bias_we, bias_re, bias_rdata, irq, dataflow=dataflow, N=N)

        common = [rst_n, go, dim_m, dim_k, dim_n, busy, done, cycles, w_addr, w_re, x_addr, x_re,
                  w_mask, x_mask, ctl, tag]
        if dataflow == 'OS':
            drain_en, drain_row = w('drain_en'), w('drain_row', 2)
            self.controller = Controller_OS(self, 'controller', *common, drain_en, drain_row)
        else:
            s_valid, s_j, s_first, s_last = w('s_valid'), w('s_j', IW), w('s_first'), w('s_last')
            self.controller = Controller_WS(self, 'controller', *common, s_valid, s_j, s_first, s_last)

        self.weight_buffer = WeightBuffer(self, 'weight_buffer', busy, wb_addr, host_wdata, wb_we, wb_re, wb_rdata,
                                          w_addr, w_re, w_rdata)
        self.activation_buffer = ActivationBuffer(self, 'activation_buffer', busy, ab_addr, host_wdata, ab_we, ab_re,
                                                  ab_rdata, x_addr, x_re, x_rdata, y_addr, y_wdata, y_we, y_wmask)

        if dataflow == 'OS':
            acc = w('acc', N * N * AW)
            self.data_feeder = DataFeeder_OS(self, 'data_feeder', rst_n, w_rdata, x_rdata, w_mask, x_mask, ctl,
                                             w_arr, x_arr, ctl_arr)
            self.pe_array = PE_Array_OS(self, 'pe_array', rst_n, w_arr, x_arr, ctl_arr, acc, N=N)
            self.output_drain = OutputDrain_OS(self, 'output_drain', rst_n, acc, drain_en, drain_row, tag,
                                               res, res_valid, tag_out, N=N)
            post = PostProcessing_OS
        else:
            south = w('psum_south', N * AW)
            self.data_feeder = DataFeeder_WS(self, 'data_feeder', rst_n, w_rdata, x_rdata, w_mask, x_mask, ctl,
                                             w_arr, x_arr, ctl_arr)
            self.pe_array = PE_Array_WS(self, 'pe_array', rst_n, x_arr, w_arr, ctl_arr, south, N=N)
            self.output_drain = OutputDrain_WS(self, 'output_drain', rst_n, south, s_valid, s_j, s_first, s_last,
                                               tag, res, res_valid, tag_out, N=N)
            post = PostProcessing_WS

        self.post_processing = post(self, 'post_processing', rst_n, busy, res, res_valid, tag_out, relu_en,
                                    req_mult, req_shift, bias_addr, host_wdata, bias_we, bias_re, bias_rdata,
                                    y_addr, y_wdata, y_we, y_wmask)
