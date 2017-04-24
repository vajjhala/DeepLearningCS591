import tensorflow as tf
import numpy as np
import read_data
import model
import sys

def train_model_one(model_dict, dataset_generators, epoch_n, print_every):
    with model_dict['graph'].as_default(), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch_i in range(epoch_n):
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                sess.run(model_dict['train_op'], feed_dict=train_feed_dict)
                
                if iter_i % print_every == 0:
                    collect_arr = []
                    for test_batch in dataset_generators['test']:
                        test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                        to_compute = [model_dict['loss'], model_dict['accuracy']]
                        collect_arr.append(sess.run(to_compute, test_feed_dict))
                    averages = np.mean(collect_arr, axis=0)
                    fmt = (epoch_i, iter_i, ) + tuple(averages)
                    
                    print('epoch {:d} iter {:d},  loss: {:.3f}, '
                          'accuracy: {:.3f}'.format(*fmt) )

                          
def new_train_model(model_dict, dataset_generators, epoch_n, print_every, variable_list,
                    save_model=False, load_model=False, model_path = "./tmp/model.ckpt"):
    with model_dict['graph'].as_default(), tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())
        
        if load_model == True:
            print("----------Restoring-----")
            saver = tf.train.Saver(var_list=variable_list)
            saver.restore(sess, model_path)
        
        for epoch_i in range(epoch_n):
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                zip_list = list(data_batch)
                zip_list.append(True)
                train_feed_dict = dict( zip( model_dict['inputs'], zip_list) ) 
                sess.run(model_dict['train_op'], feed_dict=train_feed_dict)
                
                if iter_i % print_every == 0:
                    collect_arr = []
                    for test_batch in dataset_generators['test']:
                        zip_list2 = list(test_batch)
                        zip_list2.append(False)
                        test_feed_dict = dict( zip( model_dict['inputs'], zip_list2) ) 
                        to_compute = [model_dict['loss'], model_dict['accuracy']]
                        collect_arr.append(sess.run(to_compute, test_feed_dict))
                        
                    averages = np.mean(collect_arr, axis=0)
                    fmt = (epoch_i, iter_i, ) + tuple(averages)
                    print('iteration {:d} {:d}\t loss: {:.3f}, '
                          'accuracy: {:.3f}'.format(*fmt))
        
        if save_model == True:
            print("-----------Saving ------")
            saver = tf.train.Saver(var_list=variable_list)
            save_path = saver.save(sess, model_path)
            print ("Model saved in file: %s" % save_path)
                                      
######-----PART TWO -----####################

def test_saving():

    dataset_generators = {
        'train': read_data.svhn_dataset_generator('train', 512),
        'test': read_data.svhn_dataset_generator('test', 512)
    }

    model_dict = model.apply_classification_loss(model.cnn_modified)
    new_train_model(model_dict, dataset_generators, epoch_n=100, print_every=10, variable_list=model_dict['var_list'], save_model=True, load_model=True)
   

test_saving()

#################################################
