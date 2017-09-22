# This is an example for performing linear regression in tensorflow
# It uses the scipy optimization interface
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface


s = 1000
n = 10

x = np.random.rand(s,n)
y = np.random.rand(s)


# In[23]:



# In[75]:


tf.reset_default_graph()

X = tf.placeholder("float")
Y = tf.placeholder("float")

M = tf.get_variable("beta", [n, 1], "float", initializer=tf.zeros_initializer)



y_pred = tf.matmul(X, M )
loss = tf.reduce_sum(tf.pow(Y-y_pred, 2))

optim = ScipyOptimizerInterface(loss, [M])


# In[76]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    optim.minimize(sess, feed_dict={X:x, Y:y[:,None]})
    M_st = sess.run(M)
#     ans = sess.run(loss, feed_dict={X: x, Y: y})


# In[77]:

print(M_st.T)


# In[78]:

print(np.linalg.lstsq(x,y)[0])


print("We can see that tensorflow is giving nearly identical results to numpy.")
