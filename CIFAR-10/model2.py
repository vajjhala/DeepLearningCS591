import tensorflow as tf

def logistic_regression_map(x_):
    scope_args = {'initializer': tf.random_normal_initializer(stddev=1e-4)}
    with tf.variable_scope("weights", **scope_args):
        flattered = tf.contrib.layers.flatten(x_, scope='pool2flat')
        W = tf.get_variable('W', shape=[3072, 10])
        b = tf.get_variable('b', shape=[10])
        y_logits = tf.matmul(flattered, W) + b
    return y_logits

def cnn_expanded(x_):
    var_list = []
    conv1 = tf.layers.conv2d(   inputs=x_,
                                filters=32,  # number of filters
                                kernel_size=[5, 5],
                                padding="same",
                                name = 'conv1',
                                activation=tf.nn.relu )
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                    pool_size=[2, 2], 
                                    strides=2)  # convolution stride
    
    conv2 = tf.layers.conv2d(   inputs=pool1,
                                filters=32, # number of filters
                                kernel_size=[3, 3],
                                padding="same",
                                name = 'conv2',
                                activation=tf.nn.relu)
                
    pool2 = tf.layers.max_pooling2d(inputs=conv2, 
                                    pool_size=[2, 2], 
                                    strides=2)  # convolution stride
    
    conv3 = tf.layers.conv2d(   inputs=pool1,
                                filters=32, # number of filters
                                kernel_size=[3, 3],
                                padding="same",
                                name = 'conv3',
                                activation=tf.nn.relu)
    
    pool3 = tf.layers.max_pooling2d(inputs=conv3, 
                                    pool_size=[2, 2], 
                                    strides=2)  # convolution stride
    
    pool_flat = tf.contrib.layers.flatten(pool3, scope='pool3flat') # What is pool2flat ?
    dense = tf.layers.dense(inputs=pool_flat, units=500, activation=tf.nn.relu)
    
    for scope_name in ['conv1','conv2','conv3']:            
        with tf.variable_scope(scope_name, reuse=True):
            var_list.append( tf.get_variable('kernel') )
            var_list.append( tf.get_variable('bias') )

    logits = tf.layers.dense(inputs=dense, units=10)
    cnn = dict( zip( ['logits','var_list'], [logits, var_list] ) )
    return cnn


    
############# PART 3 ################################################
def apply_classification_loss(model_function):
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):  # use gpu:0 if on GPU
            
            x_ = tf.placeholder(tf.float32, [None, 32, 32, 3])
            y_ = tf.placeholder(tf.int32, [None])      
            
            cnn =  model_function(x_)
            y_logits = cnn['logits']
            var_list = cnn['var_list']
            
            y_dict = dict(labels=y_, logits=y_logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(**y_dict)
            cross_entropy_loss = tf.reduce_mean(losses)
            trainer = tf.train.AdamOptimizer()
            train_op = trainer.minimize(cross_entropy_loss)
            
            y_pred = tf.argmax(tf.nn.softmax(y_logits), dimension=1)
            correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    model_dict = {'graph': g, 'inputs': [x_, y_], 'train_op': train_op,
                  'accuracy': accuracy, 'loss': cross_entropy_loss, 'var_list': var_list}
    
    return model_dict
######################################################################


