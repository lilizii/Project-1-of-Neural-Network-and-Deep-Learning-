from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self):
        self.layers = [
            conv2D(1, 8, 3, stride=1, padding=1),
            BatchNorm2D(8),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            conv2D(8, 16, 3, stride=1, padding=1),
            BatchNorm2D(16),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            conv2D(16, 32, 3, stride=1, padding=1),
            BatchNorm2D(32),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Linear(32 * 3 * 3, 10),
        ]
        self._shape_before_flatten = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear) and out.ndim == 4:
                self._shape_before_flatten = out.shape
                out = out.reshape(out.shape[0], -1)
            out = layer(out)
        return out

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                grads = layer.backward(grads)
                if self._shape_before_flatten is not None and grads.ndim == 2 and grads.shape[1] == np.prod(self._shape_before_flatten[1:]):
                    grads = grads.reshape(self._shape_before_flatten)
            else:
                grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            saved = pickle.load(f)
        for layer, state in zip([l for l in self.layers if l.optimizable], saved):
            params = state['params']
            for key, value in params.items():
                layer.params[key] = value
                if hasattr(layer, key):
                    setattr(layer, key, value)
            layer.weight_decay = state.get('weight_decay', False)
            layer.weight_decay_lambda = state.get('lambda', 0.0)
        
    def save_model(self, save_path):
        saved = []
        for layer in self.layers:
            if layer.optimizable:
                saved.append(
                    {
                        'params': {k: v for k, v in layer.params.items()},
                        'weight_decay': getattr(layer, 'weight_decay', False),
                        'lambda': getattr(layer, 'weight_decay_lambda', 0.0),
                    }
                )
        with open(save_path, 'wb') as f:
            pickle.dump(saved, f)


class Model_MLPCNN(Layer):
    def __init__(self):
        self.layers = [
            conv2D(1, 8, 3, stride=1, padding=1),
            BatchNorm2D(8),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            conv2D(8, 16, 3, stride=1, padding=1),
            BatchNorm2D(16),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Linear(16 * 7 * 7, 128),
            ReLU(),
            Linear(128, 10),
        ]
        self._shape_before_flatten = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers:
            if isinstance(layer, Linear) and out.ndim == 4:
                self._shape_before_flatten = out.shape
                out = out.reshape(out.shape[0], -1)
            out = layer(out)
        return out

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                grads = layer.backward(grads)
                if self._shape_before_flatten is not None and grads.ndim == 2 and grads.shape[1] == np.prod(self._shape_before_flatten[1:]):
                    grads = grads.reshape(self._shape_before_flatten)
            else:
                grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            saved = pickle.load(f)
        for layer, state in zip([l for l in self.layers if l.optimizable], saved):
            params = state['params']
            for key, value in params.items():
                layer.params[key] = value
                if hasattr(layer, key):
                    setattr(layer, key, value)
            layer.weight_decay = state.get('weight_decay', False)
            layer.weight_decay_lambda = state.get('lambda', 0.0)

    def save_model(self, save_path):
        saved = []
        for layer in self.layers:
            if layer.optimizable:
                saved.append(
                    {
                        'params': {k: v for k, v in layer.params.items()},
                        'weight_decay': getattr(layer, 'weight_decay', False),
                        'lambda': getattr(layer, 'weight_decay_lambda', 0.0),
                    }
                )
        with open(save_path, 'wb') as f:
            pickle.dump(saved, f)
