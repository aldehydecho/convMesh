# coding: utf-8
import tensorflow as tf
import numpy as np
import time
import scipy.io as sio

class convMESH():
    def __init__(self, pointnum, neighbour, degrees, maxdegree, hiddendim, finaldim, layers, lambda1, lambda2, lr):
        
        self.pointnum = pointnum
        self.hiddendim = hiddendim
        self.maxdegree = maxdegree
        self.finaldim = finaldim
        self.layers = layers
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.inputs = tf.placeholder(tf.float32, [None, self.pointnum, 9], name = 'input_mesh')
        self.nb = tf.constant(neighbour, dtype='int32', shape=[self.pointnum, self.maxdegree], name='nb_relation')
        self.degrees = tf.constant(degrees, dtype = 'float32', shape=[self.pointnum, 1], name = 'degrees')
        self.embedding_inputs = tf.placeholder(tf.float32, [None, self.hiddendim], name = 'embedding_inputs')
        self.laplacian = tf.placeholder(tf.float32, shape =(self.pointnum, self.pointnum), name = 'geodesic_weight')

        self.n_weight = []
        self.e_weight = []

        for i in range(0, self.layers):
            if i == layers - 1:
                n, e = self.get_conv_weights(9, self.finaldim, name = 'convw'+str(i+1))
            else:
                n, e = self.get_conv_weights(9, 9, name = 'convw'+str(i))

            self.n_weight.append(n)
            self.e_weight.append(e)


        self.fcparams = tf.get_variable("weights", [self.pointnum*finaldim, self.hiddendim], tf.float32, tf.random_normal_initializer(stddev=0.02))
        self.fcparams_group = tf.transpose(tf.reshape(self.fcparams, [self.pointnum, finaldim, self.hiddendim]), perm = [2, 0, 1])

        self.selfdot = tf.reduce_sum(tf.pow(self.fcparams_group, 2.0), axis = 2)

        self.maxdimension = tf.argmax(self.selfdot, axis = 1)

        self.maxlaplacian = tf.gather(self.laplacian, self.maxdimension)

        self.laplacian_norm = self.lambda1*tf.reduce_mean(tf.reduce_sum(tf.sqrt(self.selfdot) * self.maxlaplacian, 1))
        
        self.encode, self.weights_norm = self.encoder(self.inputs, train = True)
        self.decode = self.decoder(self.encode, train = True)

        self.weights_norm = self.lambda2*self.weights_norm

        self.test_encode = self.encoder(self.inputs, train = False)
        self.test_decode = self.decoder(self.test_encode, train = False)

        self.embedding_decode = self.decoder(self.embedding_inputs, train = False)
        
        self.generation_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(self.inputs-self.decode, 2.0), [1,2]))

        self.test_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(self.inputs-self.test_decode, 2.0), [1,2]))

        self.loss = self.generation_loss + self.weights_norm + self.laplacian_norm
            
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)
        
        self.saver = tf.train.Saver(max_to_keep = None)
        
    def leaky_relu(self, input_, alpha = 0.02):
        return tf.maximum(input_, alpha*input_)

    def batch_norm_wrapper(self, inputs, name = 'batch_norm',is_training = False, decay = 0.9, epsilon = 1e-5):
        with tf.variable_scope(name) as scope:
            if is_training == True:
                scale = tf.get_variable('scale', dtype=tf.float32, trainable=True, initializer=tf.ones([inputs.get_shape()[-1]],dtype=tf.float32))  
                beta = tf.get_variable('beta', dtype=tf.float32, trainable=True, initializer=tf.zeros([inputs.get_shape()[-1]],dtype=tf.float32))
                pop_mean = tf.get_variable('overallmean',  dtype=tf.float32,trainable=False, initializer=tf.zeros([inputs.get_shape()[-1]],dtype=tf.float32))
                pop_var = tf.get_variable('overallvar',  dtype=tf.float32, trainable=False, initializer=tf.ones([inputs.get_shape()[-1]],dtype=tf.float32))
            else:
                scope.reuse_variables()
                scale = tf.get_variable('scale', dtype=tf.float32, trainable=True)
                beta = tf.get_variable('beta', dtype=tf.float32, trainable=True)
                pop_mean = tf.get_variable('overallmean', dtype=tf.float32, trainable=False)
                pop_var = tf.get_variable('overallvar', dtype=tf.float32, trainable=False)

            if is_training:
                axis = list(range(len(inputs.get_shape()) - 1))
                batch_mean, batch_var = tf.nn.moments(inputs,axis)
                train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
            else:
                return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

    def convlayer(self, input_feature, input_dim, output_dim, nb_weights, edge_weights, name = 'meshconv', training = True, special_activation = False, no_activation = False, bn = True):
        with tf.variable_scope(name) as scope:
            
            padding_feature = tf.zeros([tf.shape(input_feature)[0], 1, input_dim], tf.float32)
            
            padded_input = tf.concat([padding_feature, input_feature], 1)

            def compute_nb_feature(input_f):
                return tf.gather(input_f, self.nb)

            total_nb_feature = tf.map_fn(compute_nb_feature, padded_input)
            mean_nb_feature = tf.reduce_sum(total_nb_feature, axis = 2)/self.degrees
        
            nb_feature = tf.tensordot(mean_nb_feature, nb_weights, [[2],[0]])

            edge_bias = tf.get_variable("edge_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
            edge_feature = tf.tensordot(input_feature, edge_weights, [[2],[0]]) + edge_bias

            total_feature = edge_feature + nb_feature

            if bn == False:
                fb = total_feature
            else:
                fb = self.batch_norm_wrapper(total_feature, is_training = training)

            if no_activation == True:
                fa = fb
            elif special_activation == False:
                fa = self.leaky_relu(fb)
            else:
                fa = tf.tanh(fb)

            return fa

    def get_conv_weights(self, input_dim, output_dim, name = 'convweight'):
        with tf.variable_scope(name) as scope:
            n = tf.get_variable("nb_weights", [input_dim, output_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            e = tf.get_variable("edge_weights", [input_dim, output_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))

            return n, e

    def test_model(self, geodesic_weight):

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            start = time.time()

            test = self.conv1.eval({self.inputs: feature})
            end = time.time()

            print('%fs'%(end-start))
            return test
        
    def encoder(self, input_feature, train = True):
        with tf.variable_scope("encoder") as scope:
            if(train == False):
                scope.reuse_variables()

            prev = input_feature
            
            for i in range(0, self.layers):
                if i == self.layers - 1:
                    if self.layers == 1:
                        conv = self.convlayer(prev, 9, self.finaldim, self.n_weight[i], self.e_weight[i], name = 'conv'+str(i+1), special_activation = True, training = train, bn = False)
                    else:
                        conv = self.convlayer(prev, 9, self.finaldim, self.n_weight[i], self.e_weight[i], name = 'conv'+str(i+1), no_activation = True, training = train, bn = False)
                else:
                    prev = self.convlayer(prev, 9, 9, self.n_weight[i], self.e_weight[i], name = 'conv'+str(i+1), special_activation = True, training = train, bn = False)

            l0 = tf.reshape(conv, [tf.shape(conv)[0], self.pointnum * self.finaldim])

            l1 = tf.matmul(l0, self.fcparams)

            if train == True:
                weights_maximum = tf.reduce_max(tf.abs(l1), 0) - 5
                zeros = tf.zeros_like(weights_maximum)
                weights_norm = tf.reduce_mean(tf.maximum(weights_maximum, zeros))
                return l1, weights_norm
            else:
                return l1
            
    def decoder(self, latent_tensor, train = True):
        with tf.variable_scope("decoder") as scope:
            if(train == False):
                scope.reuse_variables()
                
            l1 = tf.matmul(latent_tensor, tf.transpose(self.fcparams))
            
            l2 = tf.reshape(l1, [tf.shape(l1)[0], self.pointnum, self.finaldim])
            
            prev = l2

            for i in range(0, self.layers):
                if i == 0:
                    conv = self.convlayer(prev, self.finaldim, 9, tf.transpose(self.n_weight[self.layers-1]), tf.transpose(self.e_weight[self.layers-1]), name = 'conv'+str(i+1), special_activation = True, training = train, bn = False)
                else:
                    conv = self.convlayer(prev, 9, 9, tf.transpose(self.n_weight[self.layers-1-i]), tf.transpose(self.e_weight[self.layers-1-i]), name = 'conv'+str(i+1), special_activation = True, training = train, bn = False)

                prev = conv

        return conv
    
    def train(self, feature, geodesic_weight, maxepoch):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            
            for epoch in range(0, maxepoch+1):

                start = time.time()
                sess.run([self.optimizer], feed_dict = {self.inputs: feature, self.laplacian : geodesic_weight})

                cost_generation, cost_norm, cost_weights = sess.run([self.test_loss, self.laplacian_norm, self.weights_norm], {self.inputs:feature, self.laplacian:geodesic_weight})
                print("Epoch: [%5d|total] generation_loss: %.8f  norm_loss: %.8f weight_loss: %.8f" % (epoch, cost_generation, cost_norm, cost_weights))

                if epoch % 200 == 0 or epoch == maxepoch:
                    self.saver.save(sess, 'convmesh-model', global_step = epoch)

                end = time.time()

                print('time: %fs'%(end-start))


    def recover_mesh(self, restore, feature, logrmin, logrmax, smin, smax):
        with tf.Session() as sess:
            self.saver.restore(sess, restore)

            recover = sess.run([self.test_decode], feed_dict = {self.inputs: feature})[0]
            rs, rlogr = recover_data(recover, logrmin, logrmax, smin, smax, self.pointnum)
            sio.savemat('recover.mat', {'RS':rs, 'RLOGR':rlogr})

        return

    def individual_dimension(self, restore, feature, logrmin, logrmax, smin, smax):
        with tf.Session() as sess:
            self.saver.restore(sess, restore)

            embedding = sess.run([self.test_encode], feed_dict = {self.inputs: feature})[0]

            min_embedding = np.amin(embedding, axis = 0)

            max_embedding = np.amax(embedding, axis = 0)

            def generate_embedding_input(_min, _max, dimension, rest):
                x = np.zeros((25, self.hiddendim)).astype('float32')

                for idx in xrange(0, self.hiddendim):
                    if idx == dimension:
                        x[:, idx] = np.linspace(_min[idx], _max[idx], num = 25)
                    else:
                        x[:, idx] = rest[idx]

                return x

            for idx in xrange(0, self.hiddendim):
                embedding_data = generate_embedding_input(min_embedding, max_embedding, idx, embedding[0, :])

                recover = sess.run([self.embedding_decode], feed_dict = {self.embedding_inputs: embedding_data})[0]
                rs, rlogr = recover_data(recover, logrmin, logrmax, smin, smax, self.pointnum)
                sio.savemat('dimension'+str(idx+1)+'.mat', {'RS':rs, 'RLOGR':rlogr})

    def synthesis(self, restore, logrmin, logrmax, smin, smax, inputweight):
        with tf.Session() as sess:
            self.saver.restore(sess, restore)

            embedding = sess.run([self.test_encode], feed_dict = {self.inputs: feature})[0]

            rest = embedding[0,:]

            min_embedding = np.amin(embedding, axis = 0).reshape((hidden_dim, 1))

            max_embedding = np.amax(embedding, axis = 0).reshape((hidden_dim, 1))

            extreme_embedding = np.concatenate((min_embedding, max_embedding), axis = 1)

            direction = sio.loadmat('maxdirection.mat')
            direction = direction['maxdirection']

            eemb = np.zeros(self.hidden_dim)

            for i in xrange(0, self.hidden_dim):
                eemb[i] = extreme_embedding[i, direction[i]- 1]

            modelnum = len(inputweight)

            x = np.zeros((modelnum, self.hidden_dim))

            x[:,:] = rest

            for i in range(0, modelnum):
                for dim, weight in inputweight[i]:
                    x[i, dim] = rest[dim] + (eemb[dim] - rest[dim]) * weight

            recover = sess.run(self.embedding_decode, feed_dict = {self.embedding_inputs: x})
            rs, rlogr = recover_data(recover, logrmin, logrmax, smin, smax, self.pointnum)
            sio.savemat('synthesis.mat', {'RS':rs, 'RLOGR':rlogr})
