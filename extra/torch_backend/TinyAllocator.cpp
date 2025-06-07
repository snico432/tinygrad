#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/PyInterpreter.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/SafePyObject.h>

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
      return deleter;
    }

    at::DataPtr allocate(size_t nbytes) override {
      void *data = malloc(sizeof(std::shared_ptr<c10::SafePyObject>));
      // Check which constructor to use,
      return {data, data, &deleter, Device(kPrivateUse1)};
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
      default_copy_data(dest, src, count);
    }
  };

  static TinyAllocator g_tiny_alloc;

  REGISTER_ALLOCATOR(kPrivateUse1, &g_tiny_alloc)
}

void link_to_tiny_tensor(const at::Tensor &pt_tensor, py::object &tiny_tensor) {
  // TODO: we have to get the dtype
  std::vector<int64_t> sizes = tiny_tensor.attr("shape").cast<std::vector<int64_t>>();

  py::list views = tiny_tensor.attr("lazydata").attr("st").attr("views");
  std::vector<int64_t> strides = views[views.size() - 1].attr("strides").cast<std::vector<int64_t>>();
  int64_t storage_offset = 0;
  for (auto& v: views) {
    storage_offset += v.attr("offset").cast<int64_t>(); // TODO: is this correct?
  }

  auto* impl = pt_tensor.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(sizes, strides);
  impl->set_storage_offset(storage_offset);
 
  auto* data = pt_tensor.data_ptr();
  data = std::make_shared<c10::SafePyObject>(tiny_tensor.release().ptr(), getPyInterpreter());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("link_to_tiny_tensor", &link_to_tiny_tensor);
}