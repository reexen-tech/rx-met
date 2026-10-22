// Exercise the standalone CPU reference's actual hidden-state update.
#include "quantize_ops_helper.h"
#include <array>
#include <iostream>

int main() {
    // z=0.5 on S_z=1/256; n=0.5 and h0=0.25 on S_h=S_n=1/128.
    // The real recurrence is h_next=(h+0.5)/2, with ties-to-even on the h grid.
    constexpr std::array<int32_t, 7> expected{48, 56, 60, 62, 63, 64, 64};
    for (int32_t zero_point : {0, 1, 8, 16, 64}) {
        GateQuantParams params{};
        params.zp_update_gate_output_ = zero_point;
        params.quant_one_in_update_gate_scale_ = 256 + zero_point;
        params.shift_new_gate_output_to_h_ = 0;
        params.shift_update_old_to_h_ = 8;
        int32_t hidden = 32;
        for (auto reference : expected) {
            hidden = computeHiddenState(128 + zero_point, 64, hidden, params);
            if (hidden != reference) {
                std::cerr << "zero_point=" << zero_point << ": expected "
                          << reference << ", got " << hidden << '\n';
                return 1;
            }
        }
    }
    return 0;
}
