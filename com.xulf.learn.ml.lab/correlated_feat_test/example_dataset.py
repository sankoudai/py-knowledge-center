import tensorflow as tf

tf.random.set_seed(100)

def gen_x1(size, mean, stddev):
    return tf.random.normal([size], mean=mean, stddev=stddev, seed=100)

def add_noise(x, stddev):
    size = tf.shape(x)
    noise = tf.random.normal(size, mean=0., stddev=stddev, seed=100)
    return x + noise


def get_dataset(size, feat_spec):
    x1_spec = feat_spec['x1']
    x1 = gen_x1(size, x1_spec['mean'], x1_spec['stddev'])

    x2_spec = feat_spec['x2']
    x2 = x2_spec['func'](x1)
    x2 = add_noise(x2, x2_spec['stddev'])

    y_spec = feat_spec['y']
    y = y_spec['func'](x1, x2)
    y = add_noise(y, y_spec['stddev'])
    y = tf.sigmoid(y)

    features  = {
        'x1': x1,
        'x2': x2
    }
    dataset = tf.data.Dataset.from_tensor_slices((features, y))
    return dataset


if __name__ == '__main__':
    feat_spec = {
        'x1':{
            'mean': 1.0,
            'stddev': 0.5
        },
        'x2' : {
            'func': lambda x : x+0.1,
            'stddev': 0.1
        },
        'y':{
            'func': lambda x1, x2: 3*x1 - 0.1*x2,
            'stddev': 0.1
        }
    }

    ds = get_dataset(10, feat_spec)
    print(ds)



