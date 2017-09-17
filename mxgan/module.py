import mxnet as mx
from . import ops
import numpy as np

class GANBaseModule(object):
    def __init__(self,
                 symbol_generator,
                 context,
                 code_shape):
        # generator
        self.modG = mx.mod.Module(symbol=symbol_generator,
                                  data_names=("code",),
                                  label_names=None,
                                  context=context)
        self.modG.bind(data_shapes=[("code", code_shape)])
        # leave the discriminator
        self.temp_outG = None
        self.temp_diffD = None
        self.temp_gradD = None
        self.context = context if isinstance(context, list) else [context]
        self.outputs_fake = None
        self.outputs_real = None
        self.temp_rbatch = mx.io.DataBatch(
            [mx.nd.zeros(code_shape, ctx=self.context[-1])], None)
        
    def init_params(self, *args, **kwargs):
        self.modG.init_params(*args, **kwargs)
        self.modD.init_params(*args, **kwargs)

    def init_optimizer(self, *args, **kwargs):
        self.modG.init_optimizer(*args, **kwargs)
        self.modD.init_optimizer(*args, **kwargs)
        
    def _save_temp_gradD(self):
        if self.temp_gradD is None:
            self.temp_gradD = [
                [grad.copyto(grad.context) for grad in grads]
                for grads in self.modD._exec_group.grad_arrays]
        else:
            for gradsr, gradsf in zip(self.modD._exec_group.grad_arrays, self.temp_gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr.copyto(gradf)

    def _add_temp_gradD(self):
        # add back saved gradient
        for gradsr, gradsf in zip(self.modD._exec_group.grad_arrays, self.temp_gradD):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf
        
class GANModule(GANBaseModule):
    def __init__(self,
                 symbol_generator,
                 symbol_encoder,
                 context,
                 data_shape,
                 code_shape,
                 pos_label=0.9):
        super(GANModule, self).__init__(
            symbol_generator, context, code_shape)
        context = context if isinstance(context, list) else [context]
        self.batch_size = data_shape[0]
        label_shape = (self.batch_size, )
        encoder = symbol_encoder
        encoder = mx.sym.FullyConnected(encoder, num_hidden=1, name="fc_dloss")
        encoder = mx.sym.LogisticRegressionOutput(encoder, name='dloss')
        self.modD = mx.mod.Module(symbol=encoder,
                                  data_names=("data",),
                                  label_names=("dloss_label",),
                                  context=context)
        self.modD.bind(data_shapes=[("data", data_shape)],
                       label_shapes=[("dloss_label", label_shape)],
                       inputs_need_grad=True)
        self.pos_label = pos_label
        self.temp_label = mx.nd.zeros(
            label_shape, ctx=context[-1])
        
    def update(self, dbatch):
        # generate fake image
        mx.random.normal(0, 1.0, out=self.temp_rbatch.data[0])
        self.modG.forward(self.temp_rbatch)
        outG = self.modG.get_outputs()
        self.temp_label[:] = 0
        self.modD.forward(mx.io.DataBatch(outG, [self.temp_label]), is_train=True)
        self.modD.backward()
        self._save_temp_gradD()
        # update generator
        self.temp_label[:] = 1
        self.modD.forward(mx.io.DataBatch(outG, [self.temp_label]), is_train=True)
        self.modD.backward()
        diffD = self.modD.get_input_grads()
        self.modG.backward(diffD)
        self.modG.update()
        self.outputs_fake = [x.copyto(x.context) for x in self.modD.get_outputs()]
        # update discriminator
        self.temp_label[:] = self.pos_label
        dbatch.label = [self.temp_label]
        self.modD.forward(dbatch, is_train=True)
        self.modD.backward()
        self._add_temp_gradD()
        self.modD.update()
        self.outputs_real = self.modD.get_outputs()
        self.temp_outG = outG
        self.temp_diffD = diffD