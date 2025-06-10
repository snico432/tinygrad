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
//C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);
// NOTE: pytorch's no-op class throws error on backwards with events/streams
// TODO: why are there events in autograd?
struct CustomNoOpDeviceGuardImpl : public c10::impl::DeviceGuardImplInterface
{
  static const DeviceType D = DeviceType::PrivateUse1;
  CustomNoOpDeviceGuardImpl() = default;
  DeviceType type() const override {
    return D;
  }
  Device exchangeDevice(Device) const override {
    return Device(D, 0); // no-op
  }
  Device getDevice() const override {
    return Device(D, 0);
  }
  void setDevice(Device) const override {
    // no-op
  }
  void uncheckedSetDevice(Device) const noexcept override {
    // no-op
  }
  Stream getStream(Device) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  Stream getDefaultStream(Device) const override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  Stream getStreamFromGlobalPool(Device, bool isHighPriority = false)
      const override {
    // no-op
    (void)isHighPriority;
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  Stream getNewStream(Device, int priority = 0) const override {
    // no-op
    (void)priority;
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }
  // Event-related functions
  void record(
      void** /*event*/,
      const Stream& /*stream*/,
      const DeviceIndex /*device_index*/,
      const EventFlag /*flag*/) const override {
    //TORCH_CHECK(false, D, " backend doesn't support events.");
  }
  void block(void* /*event*/, const Stream& /*stream*/) const override {
    //TORCH_CHECK(false, D, " backend doesn't support events.")
  }
  bool queryEvent(void* /*event*/) const override {
    //TORCH_CHECK(false, D, " backend doesn't support events.")
    return true;
  }
  void destroyEvent(void* /*event*/, const DeviceIndex /*device_index*/)
      const noexcept override {}
  // Stream-related functions
  bool queryStream(const Stream& /*stream*/) const override {
    return true;
  }
  void synchronizeStream(const Stream& /*stream*/) const override {
    // Don't wait for anything.
  }
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, CustomNoOpDeviceGuardImpl);
}
}

// namespace at {
//   namespace detail {
//     C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);
//   }
// };


// Associate the DataPtr with the data from a Tinygrad Tensor
// Read up on where and how the Allocator is actually called when
// creating a tensor on a device

// Based on https://github.com/pytorch/pytorch/blob/93012d2290ed3bf25ac3d17b348dea8f7cb65150/c10/core/CPUAllocator.cpp

// TODO: Figure out how to link tiny tensor to torch tensor DataPtr
namespace c10 {
  struct C10_API TinyAllocator final : public at::Allocator {
    TinyAllocator() = default;

    static void deleter(void* ptr) {
      auto* shared_ptr = static_cast<std::shared_ptr<c10::SafePyObject>*>(ptr);
      delete shared_ptr;
    }

    DeleterFnPtr raw_deleter() const override {
      return &deleter;
    }

    at::DataPtr allocate(size_t nbytes) override {
      auto* ptr = new std::shared_ptr<c10::SafePyObject>();
      return {ptr, ptr, &deleter, Device(kPrivateUse1)};
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
      default_copy_data(dest, src, count);
    }
  };

  static TinyAllocator g_tiny_alloc;

  REGISTER_ALLOCATOR(kPrivateUse1, &g_tiny_alloc)
}

at::Tensor wrap_tensor(py::object &tiny_tensor, c10::ScalarType dtype, c10::DeviceIndex device_index)  {
  std::vector<int64_t> sizes = tiny_tensor.attr("shape").cast<std::vector<int64_t>>();

  py::list views = tiny_tensor.attr("lazydata").attr("st").attr("views");
  std::vector<int64_t> strides = views[views.size() - 1].attr("strides").cast<std::vector<int64_t>>();
  int64_t storage_offset = 0;
  for (auto& v: views) {
    storage_offset += v.attr("offset").cast<int64_t>(); // TODO: is this correct?
  }

  auto storage = c10::Storage(
    c10::Storage::use_byte_size_t(),
    0, // arbitrary value as we dont use nbytes in allocator
    c10::GetAllocator(at::kPrivateUse1)
  );

  at::Tensor pt_tensor = at::detail::make_tensor<at::TensorImpl>(
      std::move(storage),
      at::DispatchKeySet(at::DispatchKey::PrivateUse1),
      c10::scalarTypeToTypeMeta(dtype)
  );

  auto* impl = pt_tensor.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(sizes, strides);
  impl->set_storage_offset(storage_offset);

  auto tiny_tensor_shared = std::make_shared<c10::SafePyObject>(tiny_tensor.release().ptr(), getPyInterpreter());
  auto* data = static_cast<std::shared_ptr<c10::SafePyObject>*>(impl->unsafe_storage().mutable_data());
  *data = tiny_tensor_shared;
  
  return pt_tensor;
}

py::object unwrap_tensor(const at::Tensor &pt_tensor) {
  auto* impl = pt_tensor.unsafeGetTensorImpl();
  auto* data = impl->unsafe_storage().mutable_data();
  auto* tiny_tensor = static_cast<std::shared_ptr<c10::SafePyObject>*>(data);
  return py::reinterpret_borrow<py::object>(tiny_tensor->get()->ptr(getPyInterpreter()));
}

struct OpenRegHooksInterface : public at::PrivateUse1HooksInterface {
  // NOTE: no idea what this is
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override { return true; }
};

int register_hook() {
  at::RegisterPrivateUse1HooksInterface(new OpenRegHooksInterface());
  return 0;
}
int temp_register_hook = register_hook();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wrap", &wrap_tensor);
  m.def("unwrap", &unwrap_tensor);
}
