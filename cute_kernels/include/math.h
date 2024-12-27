template <typename T>
inline T ceil_divide(const T &x, const T &d) {
    assert(x > 0 && d > 0);
    return (x / d) + (x % d != 0);
}