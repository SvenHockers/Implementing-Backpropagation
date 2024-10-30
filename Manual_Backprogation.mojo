from python import Python 
from collections import List
np = Python.import_module("numpy")

def softmax(array: List[Float16]) -> List[Float16]:
    npArray = np.array(array, dtype="float32")
    expValues = np.exp(npArray - np.max(npArray))
    softmaxValues = expValues / np.sum(expValues)
    return List[Float16](softmaxValues.astype("float16").tolist())

struct Layer:
    var weight: List[Float16]
    var bias: List[Float16]

    fn __init__(inout self, numberOfInputs: Int, numberOfNodes: Int):
        self.weight = List[Float16](np.random.rand(numberOfInputs, numberOfNodes).astype("float16").tolist())
        self.bias = List[Float16](np.random.rand(numberOfNodes).astype(float16).tolist())

    fn getLayer(inout self, arg: String) -> List[Float16]:
        if arg == "weight" or arg == "weights":
            return self.weight
        elif arg == "bias":
            return self.bias
        else:
            print(arg + " : Is not an object argument")
            return List[Float16]() # here we return an empty list (not sure how else to resolve this)

    fn compute(inout self, input: List[Float16]) -> List[Float16]:
        return List[Float16]((np.array(inputs) * np.array(self.weight) + np.array(self.bias)).astype("float16").tolist())

def main():
    # initialise the Network
    var inputLayer = Layer(1, 8)
    var hiddenLayer = Layer(8, 3)
    var outputLayer = Layer(3, 8)

    var input_args = List[Float16](0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    # Compute the network
    var output = outputLayer.compute(
        hiddenLayer.compute(
            inputLayer.compute(
                input_args
            )
        )
    )
    softmaxResults = softmax(output)
    print(softmaxResults)