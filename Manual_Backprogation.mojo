## This is my first try at Mojo so assume this to be broken. 
from Tensor import Tensor
from DType import DType
from Range import range


struct SigmoidActivation:
    @staticmethod
    fn forward(input: Tensor[DType.float32]):
        let ex = input.map(fn(x: Float) -> Float:
            exp(x)
        )
        return ex.map(fn(x: Float) -> Float:
            x / (x + 1.0)
        )

    @staticmethod
    fn backward(input: Tensor[DType.float32]):
        let sigmoid = SigmoidActivation.forward(input)
        return sigmoid.map(fn(x: Float) -> Float:
            x * (1.0 - x)
        )