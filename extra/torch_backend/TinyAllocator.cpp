#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/PyInterpreter.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/SafePyObject.h>
#include <torch/torch.h>
#include <c10/core/TensorOptions.h>
#include <torch/extension.h>
#include <torch/library.h>

// register guard
namespace at {
  namespace detail {
    C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);
  }
};

// Associate the DataPtr with the data from a Tinygrad Tensor
// Read up on where and how the Allocator is actually called when
// creating a tensor on a device

// Based on https://github.com/pytorch/pytorch/blob/93012d2290ed3bf25ac3d17b348dea8f7cb65150/c10/core/CPUAllocator.cpp

// TODO: Figure out how to link tiny tensor to torch tensor DataPtr
namespace c10 {
  struct C10_API TinyAllocator final : public at::Allocator {
    TinyAllocator() = default;

    static void deleter(void* ptr) {
      free(ptr);
    }

    DeleterFnPtr raw_deleter() const override {
      return &deleter;
    }

    at::DataPtr allocate(size_t nbytes) override {
      void *data = malloc(sizeof(std::shared_ptr<c10::SafePyObject>));
      // Check which constructor to use,
      std::cout << "Allocated memory at " << data  << std::endl;
      return {data, data, &deleter, Device(kPrivateUse1)};
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
      default_copy_data(dest, src, count);
    }
  };

  static TinyAllocator g_tiny_alloc;

  REGISTER_ALLOCATOR(kPrivateUse1, &g_tiny_alloc)
}

// at::Tensor wrap_tensor(py::object &py_obj, c10::ScalarType dtype, c10::DeviceIndex device_index)
at::Tensor wrap_tensor(py::object &tiny_tensor, c10::ScalarType dtype, c10::DeviceIndex device_index)  {
  std::vector<int64_t> sizes = tiny_tensor.attr("shape").cast<std::vector<int64_t>>();

  py::list views = tiny_tensor.attr("lazydata").attr("st").attr("views");
  std::vector<int64_t> strides = views[views.size() - 1].attr("strides").cast<std::vector<int64_t>>();
  int64_t storage_offset = 0;
  for (auto& v: views) {
    storage_offset += v.attr("offset").cast<int64_t>(); // TODO: is this correct?
  }

 // Create storage with the correct size in bytes
  auto storage = c10::Storage(
    c10::Storage::use_byte_size_t(),
    0, // arbitrary value for nbytes as we dont use the value in allocator
    c10::GetAllocator(at::kPrivateUse1),
    false
  );

  at::Tensor pt_tensor = at::detail::make_tensor<at::TensorImpl>(
      std::move(storage),
      at::DispatchKeySet(at::DispatchKey::PrivateUse1),
      c10::scalarTypeToTypeMeta(dtype)
  );

  auto* impl = pt_tensor.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(sizes, strides);
  impl->set_storage_offset(storage_offset);
  auto* data = impl->unsafe_storage().mutable_data();
  std::cout << "Data pointer: " << data << std::endl;
  auto* pyobj_ptr = static_cast<std::shared_ptr<c10::SafePyObject>*>(data);
  *pyobj_ptr = std::make_shared<c10::SafePyObject>(tiny_tensor.release().ptr(), getPyInterpreter());
  
  
  return pt_tensor;
}

py::object unwrap_tensor(const at::Tensor &pt_tensor) {
  auto* impl = pt_tensor.unsafeGetTensorImpl();
  auto* data = impl->unsafe_storage().mutable_data();
  auto* tiny_tensor = static_cast<std::shared_ptr<c10::SafePyObject>*>(data);
  return py::reinterpret_borrow<py::object>(tiny_tensor->get()->ptr(getPyInterpreter()));
}

// py::object unwrap_tensor(const at::Tensor &pt_tensor) {
//   auto* impl = pt_tensor.unsafeGetTensorImpl();
//   auto* data = impl->unsafe_storage().mutable_data();





//   auto* pyobj_ptr = static_cast<std::shared_ptr<c10::SafePyObject>*>(data);
//   PyObject* raw = pyobj_ptr->get()->ptr(getPyInterpreter());
//   Py_INCREF(raw);  // Ensure lifetime is correct
//   return py::reinterpret_borrow<py::object>(raw);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wrap", &wrap_tensor);
  m.def("unwrap", &unwrap_tensor);
}