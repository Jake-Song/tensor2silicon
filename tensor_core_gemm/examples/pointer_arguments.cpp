#include <cstddef>
#include <iostream>

void change_copy(float value) {
  value = 99.0f;
  std::cout << "inside change_copy: " << value << '\n';
}

void change_through_pointer(float* value) {
  *value = 99.0f;
}

void add_arrays(const float* a, const float* b, float* output,
                std::size_t count, float beta) {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = a[i] + beta * b[i];
  }
}

int main() {
  float value = 1.0f;

  change_copy(value);
  std::cout << "after change_copy: " << value << '\n';

  change_through_pointer(&value);
  std::cout << "after change_through_pointer: " << value << '\n';

  const float a[] = {1.0f, 2.0f, 3.0f};
  const float b[] = {10.0f, 20.0f, 30.0f};
  float output[3] = {};
  add_arrays(a, b, output, 3, 0.5f);

  std::cout << "output: ";
  for (float element : output) {
    std::cout << element << ' ';
  }
  std::cout << '\n';
}
