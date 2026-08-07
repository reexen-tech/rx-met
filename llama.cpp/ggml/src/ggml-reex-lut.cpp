#ifdef GGML_USE_REEX

#include "ggml-reex-lut.h"

bool ggml_trig_lut_initialized = false;

void ggml_init_trigonometric_lut_REEX(void) {
    ggml_trig_lut_initialized = true;
}

#endif /* GGML_USE_REEX */
