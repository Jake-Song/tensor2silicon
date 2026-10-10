#include <torch/extension.h>
using torch::Tensor;
int cublas_version();
void embedding(Tensor, Tensor, Tensor);
void norm(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, double, bool);
void swiglu(Tensor, Tensor);
void rope(Tensor, Tensor, Tensor, c10::optional<Tensor>, Tensor, Tensor, Tensor,
          bool);
void argmax(Tensor, Tensor);
void advance(Tensor);
void copy_prefix(Tensor, Tensor);
void copy_token(Tensor, Tensor);
void record_token(Tensor, Tensor, int64_t);
void linear(Tensor, Tensor, Tensor);
void prefill_attention(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);
void decode_attention(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cublas_version", &cublas_version);
  m.def("embedding", &embedding);
  m.def("norm", &norm);
  m.def("swiglu", &swiglu);
  m.def("rope", &rope);
  m.def("argmax", &argmax);
  m.def("advance", &advance);
  m.def("copy_prefix", &copy_prefix);
  m.def("copy_token", &copy_token);
  m.def("record_token", &record_token);
  m.def("linear", &linear);
  m.def("prefill_attention", &prefill_attention);
  m.def("decode_attention", &decode_attention);
}
