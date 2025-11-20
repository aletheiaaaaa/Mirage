#include <cstddef>
#include <cstdint>

namespace agon::simd {
    template<size_t N = 1>
    void add_i8(const int8_t* src0, const int8_t* src1, int8_t* dst, size_t size);

}
