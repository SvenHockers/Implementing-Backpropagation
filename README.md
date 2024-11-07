# Readme File 

 ```mermaid
 classDiagram
     class SigmoidActivation {
         +forward(input: np.array) np.array
         +backward(input: np.array) np.array
     }

     class TanhActivation {
         +forward(input: np.array) np.array
         +backward(input: np.array) np.array
     }

     class ReluActivation {
           +forward(input: np.array) np.array
           +backward(input: np.array) np.array
    }

     class ActivationFunction {
         +__init__(activation_func: str) object
     }

     class layer {
         -weigths: np.ndarray
         -bias: np.ndarray
         +__init__(numberOfInputs: int, numberOfNodes: int)
         +getLayer() np.ndarray
         +compute(inputs: np.array, activation: object) np.array
     }

     class FFN {
         -activation: object
         -learningRate: float
         -layers: list<layer>
         +__init__(dimensions: list[int], activation="sigmoid", alpha=1)
         +displayModel() void
         +forward(input: list[float]) list[float]
         +backward(input: list[float], y_true: list[float]) void
         +train(X: list[float], y: list[float], max_itterations=1000, graph_points=100) void
     }

     FFN "1" *-- "many" layer : contains
     FFN "1" *-- "1" ActivationFunction : uses
     ActivationFunction "1" *-- "1" SigmoidActivation : switches to
     ActivationFunction "1" *-- "1" TanhActivation : switches to
     ActivationFunction "1" *-- "1" ReluActivation : switches to
````