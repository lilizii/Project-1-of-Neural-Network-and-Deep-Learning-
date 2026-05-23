from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # Use He init by default for ReLU-based networks to avoid unstable logits.
        if initialize_method == np.random.normal:
            self.W = np.random.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
            self.b = np.zeros((1, out_dim))
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
            self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.matmul(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.matmul(self.input.T, grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return np.matmul(grad, self.W.T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels,))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.input = None
        self.input_pad = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        if self.padding > 0:
            X_pad = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
            )
        else:
            X_pad = X
        self.input_pad = X_pad

        n, _, h, w = X_pad.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        out = np.zeros((n, self.out_channels, out_h, out_w))

        for i in range(out_h):
            hs = i * self.stride
            he = hs + self.kernel_size
            for j in range(out_w):
                ws = j * self.stride
                we = ws + self.kernel_size
                patch = X_pad[:, None, :, hs:he, ws:we]  # [n,1,c,k,k]
                out[:, :, i, j] = np.sum(patch * self.W[None, :, :, :, :], axis=(2, 3, 4)) + self.b[None, :]
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        x_pad = self.input_pad
        n, _, h, w = x_pad.shape
        _, _, out_h, out_w = grads.shape

        dW = np.zeros_like(self.W)
        db = np.sum(grads, axis=(0, 2, 3))
        dx_pad = np.zeros_like(x_pad)

        for i in range(out_h):
            hs = i * self.stride
            he = hs + self.kernel_size
            for j in range(out_w):
                ws = j * self.stride
                we = ws + self.kernel_size
                grad_ij = grads[:, :, i, j]  # [n,out]
                patch = x_pad[:, :, hs:he, ws:we]  # [n,in,k,k]
                dW += np.sum(
                    grad_ij[:, :, None, None, None] * patch[:, None, :, :, :],
                    axis=0,
                )
                dx_pad[:, :, hs:he, ws:we] += np.sum(
                    grad_ij[:, :, None, None, None] * self.W[None, :, :, :, :],
                    axis=1,
                )

        self.grads['W'] = dW
        self.grads['b'] = db

        if self.padding > 0:
            return dx_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dx_pad
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class MaxPool2D(Layer):
    """
    Simple max-pooling layer with backward support.
    """
    def __init__(self, kernel_size=2, stride=2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.max_idx = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        n, c, h, w = X.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        out = np.zeros((n, c, out_h, out_w), dtype=X.dtype)
        self.max_idx = np.zeros((n, c, out_h, out_w, 2), dtype=np.int32)

        for i in range(out_h):
            hs = i * self.stride
            he = hs + self.kernel_size
            for j in range(out_w):
                ws = j * self.stride
                we = ws + self.kernel_size
                patch = X[:, :, hs:he, ws:we]  # [n,c,k,k]
                flat = patch.reshape(n, c, -1)
                argmax = np.argmax(flat, axis=2)
                out[:, :, i, j] = np.take_along_axis(flat, argmax[:, :, None], axis=2).squeeze(-1)
                self.max_idx[:, :, i, j, 0] = argmax // self.kernel_size
                self.max_idx[:, :, i, j, 1] = argmax % self.kernel_size
        return out

    def backward(self, grads):
        n, c, h, w = self.input.shape
        _, _, out_h, out_w = grads.shape
        dx = np.zeros_like(self.input)

        for i in range(out_h):
            hs = i * self.stride
            for j in range(out_w):
                ws = j * self.stride
                r = self.max_idx[:, :, i, j, 0]
                col = self.max_idx[:, :, i, j, 1]
                for bn in range(n):
                    for ch in range(c):
                        dx[bn, ch, hs + r[bn, ch], ws + col[bn, ch]] += grads[bn, ch, i, j]
        return dx


class BatchNorm2D(Layer):
    """
    BatchNorm over N,H,W for each channel.
    """
    def __init__(self, num_features, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        self.params = {'gamma': self.gamma, 'beta': self.beta}
        self.grads = {'gamma': None, 'beta': None}
        self.optimizable = True
        self.weight_decay = False
        self.weight_decay_lambda = 0.0

        self.x = None
        self.x_hat = None
        self.inv_std = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.x = X
        mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
        var = np.var(X, axis=(0, 2, 3), keepdims=True)
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.x_hat = (X - mean) * self.inv_std
        return self.gamma * self.x_hat + self.beta

    def backward(self, grads):
        n, c, h, w = grads.shape
        m = n * h * w

        self.grads['gamma'] = np.sum(grads * self.x_hat, axis=(0, 2, 3), keepdims=True)
        self.grads['beta'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        dxhat = grads * self.gamma
        sum_dxhat = np.sum(dxhat, axis=(0, 2, 3), keepdims=True)
        sum_dxhat_xhat = np.sum(dxhat * self.x_hat, axis=(0, 2, 3), keepdims=True)
        dx = (1.0 / m) * self.inv_std * (m * dxhat - sum_dxhat - self.x_hat * sum_dxhat_xhat)
        return dx

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.probs = None
        self.labels = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.labels = labels
        probs = softmax(predicts) if self.has_softmax else predicts
        self.probs = probs
        eps = 1e-12
        picked = probs[np.arange(labels.shape[0]), labels]
        return -np.mean(np.log(picked + eps))
    
    def backward(self):
        # first compute the grads from the loss to the input
        onehot = np.zeros_like(self.probs)
        onehot[np.arange(self.labels.shape[0]), self.labels] = 1
        self.grads = (self.probs - onehot) / self.labels.shape[0]
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
