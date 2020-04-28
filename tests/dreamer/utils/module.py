import torch
import torch.nn as nn

from dreamer.utils.module import get_parameters, FreezeParameters


def test_freeze_parameters():
    linear_module_1 = nn.Linear(4, 3)
    linear_module_2 = nn.Linear(3, 2)
    input_tensor = torch.randn(4)
    with FreezeParameters([linear_module_2]):
        output_tensor = linear_module_2(linear_module_1(input_tensor))
    assert output_tensor.grad_fn is not None

    input_tensor = torch.randn(3)
    with FreezeParameters([linear_module_2]):
        output_tensor = linear_module_2(input_tensor)
    assert output_tensor.grad_fn is None

    input_tensor = torch.randn(4)
    with FreezeParameters([linear_module_1, linear_module_2]):
        output_tensor = linear_module_2(linear_module_1(input_tensor))
    assert output_tensor.grad_fn is None

    linear_module_2.weight.requires_grad = False
    linear_module_2.bias.requires_grad = True
    with FreezeParameters([linear_module_2]):
        output_tensor = linear_module_2(linear_module_1(input_tensor))
    assert output_tensor.grad_fn is not None
    assert not linear_module_2.weight.requires_grad
    assert linear_module_2.bias.requires_grad

    with FreezeParameters([linear_module_1, linear_module_2]):
        output_tensor = linear_module_2(linear_module_1(input_tensor))
    assert output_tensor.grad_fn is None
    assert not linear_module_2.weight.requires_grad
    assert linear_module_2.bias.requires_grad


def test_get_parameters():
    linear_module_1 = nn.Linear(4, 3)
    linear_module_2 = nn.Linear(3, 2)

    params = get_parameters([linear_module_1])
    assert len(params) == 2

    params = get_parameters([linear_module_1, linear_module_2])
    assert len(params) == 4
