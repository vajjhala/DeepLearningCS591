import tensorflow as tf
import numpy as np
import read_data
import model
import sys


def new_train_model(model_dict, dataset_generators, epoch_n, print_every):
    log_dir = './tmp/tb_events'
    with model_dict['graph'].as_default(), tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True) ) as sess:
         
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)           
        merged = tf.summary.merge_all()
        

        for epoch_i in range(epoch_n):
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                _, summary = sess.run([ model_dict['train_op'], merged ], feed_dict=train_feed_dict)
                train_writer.add_summary(summary)
                if iter_i % print_every == 0:
                    collect_arr = []
                    for test_batch in dataset_generators['test']:
                        test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                        to_compute = [model_dict['loss'], model_dict['accuracy']]
                        collect_arr.append(sess.run(to_compute, test_feed_dict)) 
                        test_writer.add_summary(sess.run(merged, test_feed_dict) )
                    averages = np.mean(collect_arr, axis=0)
                    avg_tpl = tuple(averages)
                    fmt = (epoch_i, iter_i, ) + avg_tpl
                    print('iteration {:d} {:d}\t loss: {:.3f}, '
                          'accuracy: {:.3f}'.format(*fmt))
        train_writer.close()
        test_writer.close()
                                      
######-----PART 4 -----##########################

def simple_svhm():

    dataset_generators = {
        'train': read_data.svhn_dataset_generator('train', 512),
        'test': read_data.svhn_dataset_generator('test', 512) }
    
    model_dict = model.apply_classification_loss(model.cnn_modified)
    new_train_model(model_dict, dataset_generators, epoch_n=100, print_every=10)
   
simple_svhm()

#################################################


















