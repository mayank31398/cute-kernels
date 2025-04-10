#include "../dtypes.h"

namespace cute_kernels::memory {
    template <typename T>
    inline __device__ T *load_128_bits(T *array, const uint64 &index) {
        using vecT = std::conditional_t<std::is_const<T>::value, const int32_4, int32_4>;
        vecT *vector_array = reinterpret_cast<vecT *>(array);
        vecT vector_element = vector_array[index];
        T *output = reinterpret_cast<T *>(&vector_element);
        return output;
    }

    template <typename T>
    inline __device__ void store_128_bits(T *source, T *destination, const uint64 &index) {
        using vecT = std::conditional_t<std::is_const<T>::value, const int32_4, int32_4>;
        vecT *destination_vector_array = reinterpret_cast<vecT *>(destination);
        vecT source_vector = reinterpret_cast<vecT *>(&source[0])[0];
        destination_vector_array[index] = source_vector;
    }

    template <typename T>
    constexpr inline __device__ uint32 get_num_elements_for_vector_load_stores() {
        return 16 / sizeof(T);
    }
}  // namespace cute_kernels::memory
