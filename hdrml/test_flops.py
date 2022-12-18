from networks.network import WholeModel

model = WholeModel(shape=(256,256,3))




import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph


from keras_flops import get_flops

inputs = tf.keras.Input(shape=(256,256,3))
outputs = model(inputs)
model_1 = tf.keras.Model(inputs=inputs, outputs=outputs)

basic flops
1,135,699,166

hdr eye fhdr
51.4114
6.7177


def get_flops(model):
    import pdb; pdb.set_trace()
    concrete = tf.function(lambda inputs: model(inputs))
    model_inputs = model.inputs if model.inputs else [[256, 256, 3]]
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs]) for inputs in model_inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


print("The FLOPs is:{}".format(get_flops(model)) ,flush=True )