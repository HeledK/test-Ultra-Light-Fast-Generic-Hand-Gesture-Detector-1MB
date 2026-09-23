import tensorflow as tf

model = tf.keras.models.load_model(r'export_models/slim_ultraslim')

for layer in model.layers:
    t = type(layer).__name__
    if t in {'Activation', 'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'Softmax'}:
        print(f'{layer.name}: {t}')
    # Also check layers that have an `activation` attribute (Conv2D, Dense often bake it in)
    if hasattr(layer, 'activation') and layer.activation is not None:
        act_name = layer.activation.__name__ if callable(layer.activation) else str(layer.activation)
        if act_name != 'linear':
            print(f'{layer.name} ({t}): activation={act_name}')