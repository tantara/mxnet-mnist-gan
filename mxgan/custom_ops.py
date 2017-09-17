import numpy as np
import mxnet as mx
import pickle as pkl

@mx.operator.register("constant")
class ConstantOpProp(mx.operator.CustomOpProp):
    def __init__(self, pkl_data):
        super(ConstantOpProp, self).__init__(need_top_grad=False)
        self.data = pkl.loads(pkl_data)

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [self.data.shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return ConstantOp(mx.nd.array(self.data))

def constant(data, name="constant"):
    if isinstance(data, mx.nd.NDArray):
        data = data.asnumpy()
    pkl_data = pkl.dumps(data)
    return mx.symbol.Custom(name=name,
                            op_type="constant",
                            pkl_data=pkl_data)