#include <iostream>
using namespace std;

template <typename T, typename U>
// T max(T x, T y) {
//     return (x > y) ? x : y;
// }

auto max(T x, U y) {
    return (x > y) ? x : y;
}

int main() {
    std::cout << max(1, 2) << endl;
    std::cout << max(3.1, 2.9) << endl;
    std::cout << max(3.1, 2) << endl;
    return 0;
}