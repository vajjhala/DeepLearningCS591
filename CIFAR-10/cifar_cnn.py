import tensorflow as tf
import numpy as np
import read_data
import model2
import sys
import read_cifar10 as cf10


def new_train_model(model_dict, dataset_generators, epoch_n, print_every, variable_list,
                    save_model=False, load_model=False, model_path = "./tmp/model_full.ckpt"):
    with model_dict['graph'].as_default(), tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())
        # To make sure everyone gets initialised
        
        if load_model == True:
            print("----------Restoring--------")
            saver = tf.train.Saver(var_list=variable_list)
            saver.restore(sess, model_path) 
            # restore automatically initialises
        
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
                    print('iteration {:d} {:d}\t loss: {:.3f}, '
                          'accuracy: {:.3f}'.format(*fmt))
        
        if save_model == True:
            print("-----------Saving ------")
            saver = tf.train.Saver(var_list=variable_list)
            save_path = saver.save(sess, model_path)
            print ("Model saved in file: %s" % save_path)
                                      



######---------####################
def cifar_main():
    print("####----------CIFAR on CNN---------------#####") 
    cifar10_dataset_generators = {
            'train': read_data.cifar10_dataset_generator('train', 1000),
            'test': read_data.cifar10_dataset_generator('test', -1)    }

    ## train a model from scratch
    cnn_expanded_dict = model2.apply_classification_loss(model2.cnn_expanded)

    new_train_model(cnn_expanded_dict, cifar10_dataset_generators, epoch_n=100, 
                print_every=10, variable_list=cnn_expanded_dict['var_list'], save_model=False)
   
######----------####################
def svhn_tuner():

    cifar10_dataset_generators = {
            'train': read_data.cifar10_dataset_generator('train', 1000),
            'test': read_data.cifar10_dataset_generator('test', -1)    }
            
    dataset_generators = {
        'train': read_data.svhn_dataset_generator('train', 512),
        'test': read_data.svhn_dataset_generator('test', 512) }
    
    cnn_expanded_dict = model2.apply_classification_loss(model2.cnn_expanded)
    
    
    print("####----- Running CNN_Expanded on SVHN and saving ------####")
    new_train_model(cnn_expanded_dict, dataset_generators, epoch_n=50, print_every=10, 
                        variable_list=cnn_expanded_dict['var_list'], save_model=True ) 
    
    print("#### --------- Loading saved weights into new input ------ ####") 
    new_train_model(cnn_expanded_dict, cifar10_dataset_generators, epoch_n=100, 
                     print_every=10, variable_list=cnn_expanded_dict['var_list'], load_model=True)
    

    
#################################################
   
cifar_main()  
svhn_tuner()

