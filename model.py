import tensorflow as tf
import tensorflow_probability as tfp 
tfd = tfp.distributions
tfb = tfp.bijectors
import time
import numpy as np

class Invicuna:
    # currently not using conditional density estimation in latent model
    # current implementation assumes known variance; no conditional density est.
    
    def __init__(self, sess, config):
        self.config = config #should this be deepcopied?
        self.sess = sess
        
        self.y_dim = config['y_dim']
        
        
        # define prior values
        # TODO move these inits to config
        self.mu_0 = np.zeros([self.y_dim]) # ([[0.],[0.]])
        self.sig_0 = 1. #covariance of gaussian
        self.updates_so_far = 0
        self.S0_reg_const = 0.1
        
    def construct_model(self):
        with self.sess.graph.as_default():
            
            #build priors
            if self.sig_0 is list:
                raise ValueError('need to define inits for this case')
            else:
                # self.V0_inv = (1./self.sig_0)*tf.eye(self.x_dim)
                self.V0_asym = tf.get_variable('V0_asym',initializer=1/self.sig_0*tf.eye(self.y_dim))
                self.V0_inv = self.V0_asym @ tf.transpose(self.V0_asym) #
                self.V0 = tf.linalg.inv(self.V0_inv)
                
                # making S0 trainable; enforce invertibility via cholesky
                # self.S0_asym = tf.Variable(5*self.sig_0*tf.eye(self.x_dim)) # cholesky decomp of \Lambda_0
                # self.S0_inv = self.S0_asym @ tf.transpose(self.S0_asym)
                self.S0_inv = self.sig_0*tf.eye(self.y_dim)
                self.S0 = tf.linalg.inv(self.S0_inv)
    
    
            self.mu_0 = tf.constant(self.mu_0, dtype=tf.float32)
            
            # context_x,y: x,y points available for context (M, N_context, x_dim/y_dim)
            self.context_y = tf.placeholder(tf.float32, shape=[None,None,self.y_dim], name="cy")
            
            # y: query points (M, N_test, x_dim)
            self.y = tf.placeholder(tf.float32, shape=[None,None,self.y_dim], name="y")
            
            # build network
            self.flow_bijector = self.build_flow()
        
            # num_updates: number of context points from context_x,y to use when computing posterior. size (M,)
            self.num_models = tf.shape(self.context_y)[0]
            self.max_num_context = tf.shape(self.context_y)[1]*tf.ones((self.num_models,), dtype=tf.int32)
            self.num_context = tf.placeholder_with_default(self.max_num_context, shape=(None,))

            # in the case of conditional density est, map x to feature space
            
            # map context data to latent space
            # self.context_phi is (M, N_context, phi_dim)
            self.context_z = tf.map_fn( lambda y: self.flow_bijector.inverse(y),
                                        elems=self.context_y,
                                        dtype=tf.float32)
            
            # compute posteriors
            self.mu_N, self.V_N = tf.map_fn(lambda x: self.batch_gaussian_update(*x),
                                                            elems=(self.context_z, self.num_context),
                                                            dtype=(tf.float32, tf.float32) 
                                                            )
            
            
            # posterior base distribution
            self.base = tfd.MultivariateNormalFullCovariance(loc=self.mu_N,                     covariance_matrix=self.V_N + self.S0)
                        
            self.transformed_dist = tfd.TransformedDistribution(distribution=self.base,bijector=self.flow_bijector)
            
            y_transposed = tf.transpose(self.y, perm=[1,0,2])
            self.loss = -(self.transformed_dist.log_prob(y_transposed)) 
            self.total_loss = tf.reduce_mean(self.loss)
            
            optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
            gs, vs = zip(*optimizer.compute_gradients(self.total_loss))
            # v_names = [v.name for v in tf.trainable_variables()]
            # global_norms = [tf.reduce_max(g) for g in gs]
            #print_op = tf.print(list(zip(v_names,global_norms)))
            
            #with tf.control_dependencies([print_op]):
                # gs, _ = tf.clip_by_global_norm(gs, 5.)
            self.train_op = optimizer.apply_gradients(zip(gs, vs))
            
            
            
            
            # rmse_z = tf.reduce_mean( tf.sqrt( tf.reduce_sum( (self.z - tf.expand_dims(self.mu_N, axis=1))**2, axis=-1) ) )
            # tf.summary.scalar("rmse_z", rmse_z)
            
            norm_S0_inv = tf.reduce_mean( tf.norm(self.S0_inv, ord='fro', axis=(-2,-1)) )
            tf.summary.scalar("norm_S0_inv", norm_S0_inv)
            
            norm_V0_inv = tf.reduce_mean( tf.norm(self.V0_inv, ord='fro', axis=(-2,-1)) )
            tf.summary.scalar("norm_V0_inv", norm_V0_inv)
            
            # mean_invJ_logdet = tf.reduce_mean( logdetinvJ )
            # tf.summary.scalar("mean_invJ_logdet", mean_invJ_logdet)
                        
            self.train_writer = tf.summary.FileWriter('summaries/'+str(time.time()), self.sess.graph, flush_secs=10)
            self.merged = tf.summary.merge_all()

            self.saver = tf.train.Saver()
                        
            self.sess.run(tf.global_variables_initializer())
        
    
    def build_flow(self):
        use_batchnorm = self.config['use_batchnorm']
        net_size = self.config['network_size']
        
        
        bijectors = []
        if self.config['bijector'] == 'RealNVP':
            bij = tfb.RealNVP(
                      num_masked=int(self.y_dim / 2),
                      shift_and_log_scale_fn=tfb.real_nvp_default_template(
                          hidden_layers=[net_size, net_size]
                      ))
        else:
            raise ValueError('invalid bijector in config')



        for i in range(self.config['num_bijectors']):
            bijectors.append(bij)

            if use_batchnorm and i%4==2:
                # This appears to break everything when used; at least for non-centered distribs
                bijectors.append(tfb.BatchNormalization())

            bijectors.append(tfb.Permute(permutation=np.random.permutation(self.y_dim)))


        if self.config['num_bijectors'] > 0:
            return tfb.Chain(list(reversed(bijectors[:-1]))) # discard last permutation
        else:
            return tfb.Identity()
        
    def batch_gaussian_update(self,Y,num):
        Y = Y[:num,:]
        numf = tf.to_float(num)
        V_N = tf.matrix_inverse( self.V0_inv + numf * self.S0_inv )
        y_sum = 0.*self.mu_0 + tf.reduce_sum(Y, axis=0, keepdims=True)
        t1 = self.S0_inv @ tf.matrix_transpose(y_sum)
        t2 = self.V0_inv @ tf.expand_dims(self.mu_0, axis=-1)
        m_N = tf.squeeze( V_N @ (t1 + t2), axis=-1 )
        return m_N, V_N
        
    def train(self,dataset,num_train_updates):
        batch_size = self.config['batch_size']
        horizon = self.config['data_horizon']
        test_horizon = self.config['test_horizon']

        #minimize loss
        for i in range(num_train_updates):
            y = dataset.sample(n_funcs=batch_size, n_samples=horizon+test_horizon)
            feed_dict = {
                    self.context_y: y[:,:horizon,:],
                    self.y: y[:,horizon:,:],
                    self.num_context: np.random.randint(horizon+1,size=batch_size)
                    }
            loss, _, summary = self.sess.run([self.total_loss,self.train_op,self.merged],feed_dict)
        
            if i % 20 == 0:
                print('itr:', str(i), '; loss:',loss)

            self.train_writer.add_summary(summary, self.updates_so_far)
            self.updates_so_far += 1
    
    def gen_samples(self,cond_y,num_samples):
        feed_dict ={self.context_y: cond_y}
        samples = self.sess.run(self.transformed_dist.sample(num_samples), feed_dict)
        return samples
        
    def get_likelihood(self,cond_y,query_points):
        feed_dict ={self.context_y: cond_y,
                    self.y: query_points}
        probs = np.exp(-self.sess.run(self.loss, feed_dict))
        return probs
    
class CondChain(tfb.Chain, tfb.ConditionalBijector):
    pass

class CondMVN(tfd.MultivariateNormalFullCovariance):
    def _log_prob(self, value, name, **condition_kwargs):
        return MultivariateNormalFullCovariance._log_prob(self, value, name)
    
class CondIdentity(tfb.Identity, tfb.ConditionalBijector):
    pass

class CondInvicuna:
    # current implementation assumes known variance
    
    def __init__(self, sess, config):
        self.config = config
        self.sess = sess
        
        self.x_dim = config['x_dim']
        self.phi_dim = config['nn_layers'][-1]
        self.y_dim = config['y_dim']
        self.sigma_eps = self.config['sigma_eps']
        
        self.M = config['batch_size']
        
        # define prior values
        # TODO move these inits to config
        self.mu_0 = np.zeros([self.y_dim]) # ([[0.],[0.]])
        self.sig_0 = 1. #covariance of gaussian
        self.updates_so_far = 0
        self.S0_reg_const = 0.1
        
    def construct_model(self):
        with self.sess.graph.as_default():
            last_layer = self.config['nn_layers'][-1]
                
            # build priors
            self.SigEps = self.sigma_eps*tf.eye(self.y_dim)
            self.SigEps = tf.reshape(self.SigEps, (1,1,self.y_dim,self.y_dim))
            
            self.K = tf.get_variable('K_init',shape=[last_layer,self.y_dim]) #\bar{K}_0

            self.L_asym = tf.get_variable('L_asym',initializer=tf.eye(last_layer)) # cholesky decomp of \Lambda_0
            self.L = self.L_asym @ tf.matrix_transpose(self.L_asym) # \Lambda_0
    
            
            # context_x,y: x,y points available for context (M, N_context, x_dim/y_dim)
            self.context_x = tf.placeholder(tf.float32, shape=[self.M,None,self.x_dim], name="cx")
            self.context_y = tf.placeholder(tf.float32, shape=[self.M,None,self.y_dim], name="cy")
            
            # y: query points (M, N_test, x_dim)
            self.x = tf.placeholder(tf.float32, shape=[self.M,None,self.x_dim], name="x")
            self.y = tf.placeholder(tf.float32, shape=[self.M,None,self.y_dim], name="y")
            
            # encode x to phi(x)
            self.context_phi = tf.map_fn( lambda x: self.basis(x),
                                          elems=self.context_x,
                                          dtype=tf.float32)
            self.phi = tf.map_fn( lambda x: self.basis(x),
                                  elems=self.x,
                                  dtype=tf.float32)
            
            # build invertible flow network
            self.flow_bijector = self.build_flow()
        
            # num_updates: number of context points from context_x,y to use when computing posterior. size (M,)
            self.num_models = tf.shape(self.context_y)[0]
            self.max_num_context = tf.shape(self.context_y)[1]*tf.ones((self.num_models,), dtype=tf.int32)
            self.num_context = tf.placeholder_with_default(self.max_num_context, shape=(None,))

            # in the case of conditional density est, map x to feature space
            
            # map context y to latent space
            # self.context_z is (M, N_context, y_dim)
            self.context_z = tf.map_fn( lambda xy: self.flow_bijector.inverse(xy[1], x=xy[0]),
                                        elems=(self.context_x,self.context_y),
                                        dtype=tf.float32)
            
            # compute posteriors
            self.K_N, self.Linv_N = tf.map_fn(lambda x: self.batch_blr(*x),
                                                            elems=(self.context_phi, self.context_y, self.num_context),
                                                            dtype=(tf.float32, tf.float32) 
                                                            )
            
            # compute posterior predictive in latent space
            self.mu_N = batch_matmul(tf.matrix_transpose(self.K_N), self.phi)
            spread_fac = 1 + batch_quadform(self.Linv_N, self.phi)
            self.Sig_N = tf.expand_dims(spread_fac, axis=-1)*self.SigEps

            
#             print_op = tf.print(tf.reduce_mean(self.Sig_N, axis=(0,1)), tf.linalg.det(self.Linv_N), tf.linalg.det(tf.linalg.inv(self.L)))
#             with tf.control_dependencies([print_op]):
            self.base = tfd.MultivariateNormalFullCovariance(loc=self.mu_N, covariance_matrix=self.Sig_N)
            
            # map test data to latent space to evaluate log likelihood
            self.z = tf.map_fn( lambda xy: self.flow_bijector.inverse(xy[1], x=xy[0]),
                                        elems=(self.x,self.y),
                                        dtype=tf.float32)
            
            rmse_z = tf.reduce_mean( tf.sqrt( tf.reduce_sum( (self.z - tf.expand_dims(self.mu_N, axis=1))**2, axis=-1) ) )
            tf.summary.scalar("rmse_z", rmse_z)
            
            logdetinvJ = tf.map_fn( lambda xy: self.flow_bijector.inverse_log_det_jacobian(xy[1], event_ndims=1, x=xy[0]),
                                      elems=(self.x, self.y),
                                      dtype=tf.float32)
            
            self.loss = -self.base.log_prob(self.z) -logdetinvJ
            
            # map to observation space
            #self.transformed_dist = tfd.ConditionalTransformedDistribution(distribution=self.base,bijector=self.flow_bijector)
            
            
            #self.loss = -(self.transformed_dist.log_prob(self.y, x=self.x)) 
            self.total_loss = tf.reduce_mean(self.loss)
            tf.summary.scalar("loss", self.total_loss)
            
            optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
            gs, vs = zip(*optimizer.compute_gradients(self.total_loss))
            gs, _ = tf.clip_by_global_norm(gs, 5.)
            self.train_op = optimizer.apply_gradients(zip(gs, vs)) #minimize(self.total_loss)#
                        
            self.train_writer = tf.summary.FileWriter('summaries/'+str(time.time()), self.sess.graph, flush_secs=10)
            self.merged = tf.summary.merge_all()

            self.saver = tf.train.Saver()
                        
            self.sess.run(tf.global_variables_initializer())
    
    def build_flow(self):
        use_batchnorm = self.config['use_batchnorm']
        net_size = self.config['network_size']
        
        
        bijectors = []        
        for i in range(self.config['num_bijectors']):
            bij = tfb.RealNVP(
                      num_masked=int(self.y_dim / 2),
                      shift_and_log_scale_fn=real_nvp_conditional_template(
                          hidden_layers=[net_size, net_size]
                      ))
            
            bijectors.append(bij)

            if use_batchnorm and i%4==2:
                # This appears to break everything when used; at least for non-centered distribs
                bijectors.append(tfb.BatchNormalization())

            bijectors.append(tfb.Permute(permutation=np.random.permutation(self.y_dim)))


        if self.config['num_bijectors'] > 0:
            return CondChain(list(reversed(bijectors[:-1]))) # discard last permutation
        else:
            return CondIdentity(validate_args=False)
        
    def basis(self,x,name='basis'):
        layer_sizes = self.config['nn_layers']
        activations = {
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid
        }
        activation = activations[ self.config['activation'] ]

        inp = x
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            for units in layer_sizes:
                inp = tf.layers.dense(inputs=inp, units=units,activation=activation)

        return inp
    
    def batch_blr(self,X,Y,num):
        X = X[:num,:]
        Y = Y[:num,:]
        Ln_inv = tf.matrix_inverse(tf.transpose(X) @ X + self.L)
        Kn = Ln_inv @ (tf.transpose(X) @ Y + self.L @ self.K)
        return tf.cond( num > 0, lambda : (Kn,Ln_inv), lambda : (self.K, tf.linalg.inv(self.L)) )
        
    def train(self,dataset,num_train_updates):
        batch_size = self.config['batch_size']
        horizon = self.config['data_horizon']
        test_horizon = self.config['test_horizon']

        #minimize loss
        for i in range(num_train_updates):
            x,y = dataset.sample(n_funcs=batch_size, n_samples=horizon+test_horizon)
            feed_dict = {
                    self.context_x: x[:,:horizon,:],
                    self.context_y: y[:,:horizon,:],
                    self.x: x[:,horizon:,:], 
                    self.y: y[:,horizon:,:],
                    self.num_context: np.random.randint(horizon+1,size=batch_size)
                    }
            loss, _, summary = self.sess.run([self.total_loss,self.train_op,self.merged],feed_dict)
        
            if i % 20 == 0:
                print('itr:', str(i), '; loss:',loss)

            self.train_writer.add_summary(summary, self.updates_so_far)
            self.updates_so_far += 1
    
    def gen_samples(self,cond_x,cond_y,x,num_samples):
        feed_dict ={self.context_y: cond_y,
                    self.context_x: cond_x,
                    self.x: x}
        samples = self.sess.run(self.transformed_dist.sample(num_samples), feed_dict)
        return samples
        
    def get_likelihood(self,cond_x,cond_y,x,y):
        feed_dict ={self.context_y: cond_y,
                    self.context_x: cond_x,
                    self.x: x,
                    self.y: y}
        probs = np.exp(-self.sess.run(self.loss, feed_dict))
        return probs
    
def real_nvp_conditional_template(
    hidden_layers,
    shift_only=False,
    activation=tf.nn.relu,
    name=None,
    *args,
    **kwargs):

    with tf.name_scope(name, "real_nvp_cond_template"):
        def _fn(x, output_units, **condition_kwargs):
            """Fully connected MLP parameterized via `real_nvp_template`."""
            if condition_kwargs:
                x = tf.concat([x, condition_kwargs['x']], axis=-1)
            for units in hidden_layers:
                x = tf.layers.dense(
                    inputs=x,
                    units=units,
                    activation=activation,
                    *args,
                    **kwargs)
            x = tf.layers.dense(
              inputs=x,
              units=(1 if shift_only else 2) * output_units,
              activation=None,
              *args,
              **kwargs)
            if shift_only:
                return x, None
            shift, log_scale = tf.split(x, 2, axis=-1)
            return shift, log_scale

        return tf.make_template("real_nvp_default_template", _fn)
    
# given mat [a,b,c,...,N,N] and batch_v [a,b,c,...,M,N], returns [a,b,c,...,M,N]
def batch_matmul(mat, batch_v, name='batch_matmul'):
    with tf.name_scope(name):
        return tf.matrix_transpose(tf.matmul(mat,tf.matrix_transpose(batch_v)))

# works for A = [...,n,n] or [...,N,n,n]
# (uses the same matrix A for all N b vectors in the first case)
# assumes b = [...,N,n]
# returns  [...,N,1]
def batch_quadform(A, b):
    A_dims = A.get_shape().ndims
    b_dims = b.get_shape().ndims
    b_vec = tf.expand_dims(b, axis=-1)
    if A_dims == b_dims + 1:
        return tf.squeeze( tf.matrix_transpose(b_vec) @ A @ b_vec, axis=-1)
    elif A_dims == b_dims:
        Ab = tf.expand_dims( tf.matrix_transpose( A @ tf.matrix_transpose(b) ), axis=-1) # ... x N x n x 1
        return tf.squeeze( tf.matrix_transpose(b_vec) @ Ab, axis = -1) # ... x N x 1
    else:
        raise ValueError('Matrix size of %d is not supported.'%(A_dims))