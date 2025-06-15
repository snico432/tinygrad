from tinygrad.dtype import _from_torch_dtype, _to_torch_dtype
import torch
import tinygrad
import torch.utils._pytree as torch_pytree

class Tensor(torch.Tensor):

  @staticmethod
  def __new__(cls, elem, env):
    dtype = _to_torch_dtype(elem.dtype)
    shape = list(elem.shape)
    for i, s in enumerate(shape):
      if not isinstance(s, int):
        shape[i] = 1
    if dtype is None:
      dtype = torch.float32
    return torch.Tensor._make_wrapper_subclass(
        cls,
        shape,
        dtype=dtype,
        device="meta",
        requires_grad=False,
    )

  def __init__(self, elem: tinygrad.Tensor, env: "Environment"):
    super().__init__()
    self._elem = elem
    self._env = env

  def __str__(self):
    return "Tensor({} {})".format(str(type(self._elem)), str(self._elem))

  __repr__ = __str__

  def __jax_array__(self):
    return self._elem

  @property
  def shape(self):
    return self._elem.shape

  @property
  def ndim(self):
    return len(self._elem.shape)

  def flatten(self, start_dim=0, end_dim=-1):
    if end_dim == -1:
      end_dim = self.ndim
    new_shape = (
        self._elem.shape[:start_dim] + (-1,) + self._elem.shape[end_dim + 1:])
    new_elem = self._elem.reshape(new_shape)
    return Tensor(new_elem, self._env)

  __torch_function__ = torch._C._disabled_torch_function_impl

  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    env = None
    for arg in torch_pytree.arg_tree_leaves(*args, **kwargs):
      if isinstance(arg, Tensor):
        env = arg._env
        break

    with env:
      return func(*args, **(kwargs or {}))

  def detach(self):
    return Tensor(self.tiny().detach(), self._env)

  def tiny(self):
    return self._elem

  @property
  def dtype(self):
    return _from_torch_dtype(self._elem.dtype)

  def dim(self):
    return self.ndim
