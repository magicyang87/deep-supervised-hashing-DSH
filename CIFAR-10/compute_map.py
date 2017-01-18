
# coding: utf-8

# In[109]:

import numpy as np


# In[110]:

code_path = '/home/dldev/github/deep-supervised-hashing-DSH/CIFAR-10/code.dat'
label_path = '/home/dldev/github/deep-supervised-hashing-DSH/CIFAR-10/label.dat'

codes = np.fromfile(code_path, np.float32)
labels = np.fromfile(label_path, np.float32)
label_num = len(labels)
codes = codes.reshape([label_num, len(codes) / label_num])


# In[111]:

codes = np.sign(codes)
dist = -codes.dot(codes.T)


# In[112]:

sim_idx_mat = np.argsort(dist, axis=0, kind='mergesort')


# In[113]:

sorted_database_label_mtx = np.mat(labels[sim_idx_mat])


# In[114]:

label_rep_mat = np.repeat(np.mat(labels), label_num, axis=0)


# In[115]:

result_mat = (sorted_database_label_mtx == label_rep_mat)


# In[116]:

precision_list = []
for i in xrange(result_mat.shape[1]):
    Qi = result_mat[:,i].sum()
    true_arr = result_mat[:,i].nonzero()[0]
    precision_list.append(np.divide(np.arange(1,Qi+1)*1.0, true_arr+1).sum() / Qi)
map = np.mean(precision_list)

print 'mAP: ', map

