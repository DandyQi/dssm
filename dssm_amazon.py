import pickle
import random
import time
import sys
import numpy as np
import tensorflow as tf
from scipy import sparse
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', '/tmp/dssm-400-120-relu', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 10, "Number of steps in one epoch.")
flags.DEFINE_integer('pack_size', 10, "Number of batches in one pickle pack.")
flags.DEFINE_bool('gpu', 0, "Enable GPU or not")
flags.DEFINE_integer('epochs',50,'Number of epochs')
start = time.time()

class DataSet:
	def __init__(self):
		self.train_query = sparse.csr_matrix(pickle.load(open("./feature/train_u.pkl", "rb")))
		self.train_pos = sparse.csr_matrix(pickle.load(open("./feature/train_i.pkl", "rb")))
		self.eval_query = sparse.csr_matrix(pickle.load(open("./feature/test_u.pkl", "rb")))
		self.eval_pos = sparse.csr_matrix(pickle.load(open("./feature/test_i.pkl", "rb")))

data_sets = DataSet()
NEG = 10
BS = 30
# TRIGRAM_D = 15644
input_u_D = data_sets.train_query.shape[1]#11402
input_i_D = data_sets.train_pos.shape[1]#7891
L1_N = 400
L2_N = 120
# 是否加BN层
# norm, epsilon = True, 0.001

query_in_shape = np.array([BS, input_u_D])
doc_in_shape = np.array([BS, input_i_D])

with tf.name_scope('input'):
	# Shape [BS, TRIGRAM_D].
	query_batch = tf.sparse_placeholder(tf.float32, shape=query_in_shape, name='QueryBatch')
	# Shape [BS, TRIGRAM_D]
	doc_batch = tf.sparse_placeholder(tf.float32, shape=doc_in_shape, name='DocBatch')

with tf.name_scope('L1'):
	l1_par_range = np.sqrt(6.0 / (input_u_D + L1_N))
	weight1_q = tf.Variable(tf.random_uniform([input_u_D, L1_N], -l1_par_range, l1_par_range))
	bias1_q = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))
	weight1_d = tf.Variable(tf.random_uniform([input_i_D, L1_N], -l1_par_range, l1_par_range))
	bias1_d = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))

	query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1_q) + bias1_q

	doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, weight1_d) + bias1_d
	query_l1_out = tf.nn.relu(query_l1)
	doc_l1_out = tf.nn.relu(doc_l1)

with tf.name_scope('L2'):
	l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))

	weight2_q = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
	bias2_q = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
	weight2_d = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
	bias2_d = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
	query_l2 = tf.matmul(query_l1_out, weight2_q) + bias2_q
	doc_l2 = tf.matmul(doc_l1_out, weight2_d) + bias2_d
	query_y = tf.nn.relu(query_l2)
	doc_y = tf.nn.relu(doc_l2)

with tf.name_scope('FD_rotate'):
	temp = tf.tile(doc_y, [1, 1])

	for i in range(NEG):
		rand = int((random.random() + i) * BS / NEG)
		doc_y = tf.concat([doc_y,
						   tf.slice(temp, [rand, 0], [BS - rand, -1]),
						   tf.slice(temp, [0, 0], [rand, -1])], axis = 0)

with tf.name_scope('Cosine_Similarity'):
	# Cosine similarity
	query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
	doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

	prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
	norm_prod = tf.multiply(query_norm, doc_norm)

	cos_sim_raw = tf.truediv(prod, norm_prod)
	cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * 20

with tf.name_scope('Loss'):
	# Train Loss
	prob = tf.nn.softmax((cos_sim))
	hit_prob = tf.slice(prob, [0, 0], [-1, 1])
	loss = -tf.reduce_sum(tf.log(hit_prob))     / BS

with tf.name_scope('Training'):
	# Optimizer
	train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

with tf.name_scope('Accuracy'):
	correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# tf.summary.scalar('accuracy', accuracy)

def pull_batch(query_data, doc_data, batch_idx):
	# start = time.time()
	query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
	doc_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
	query_in = query_in.tocoo()
	doc_in = doc_in.tocoo()

	query_in = tf.SparseTensorValue(
		np.transpose([np.array(query_in.row, dtype=np.float), np.array(query_in.col, dtype=np.float)]),
		np.array(query_in.data, dtype=np.float),
		np.array(query_in.shape, dtype=np.int64))
	doc_in = tf.SparseTensorValue(
		np.transpose([np.array(doc_in.row, dtype=np.float), np.array(doc_in.col, dtype=np.float)]),
		np.array(doc_in.data, dtype=np.float),
		np.array(doc_in.shape, dtype=np.int64))
	return query_in, doc_in

def feed_dict(Train, batch_idx):
	"""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
	if Train:
		# batch_idx = int(random.random() * (FLAGS.epoch_steps - 1))
		query_in, doc_in = pull_batch(data_sets.train_query, data_sets.train_pos, batch_idx)
	else:
		# batch_idx = int(random.random() * (FLAGS.epoch_steps - 1))
		query_in, doc_in = pull_batch(data_sets.eval_query, data_sets.eval_pos, batch_idx)
	return {query_batch: query_in, doc_batch: doc_in}


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# Actual execution

	start = time.time()
	for epoch in range(FLAGS.epochs):
		for i in range(data_sets.train_pos.shape[0]//BS):
			# batch_idx = step % FLAGS.epoch_steps
			sess.run(train_step, feed_dict=feed_dict(True, i))#batch_idx % FLAGS.pack_size))

			if i == 0:
				end = time.time()
				epoch_loss = 0
				acc = 0
				s = data_sets.train_pos.shape[0]//BS
				for j in range(s):
					loss_v = sess.run(loss, feed_dict=feed_dict(True, j))
					# print('query_norm',sess.run(query_norm, feed_dict=feed_dict(False, True, i, 1)))
					# print('doc_norm',sess.run(doc_norm, feed_dict=feed_dict(False, True, i, 1)))
					# print('norm_prod',sess.run(norm_prod, feed_dict=feed_dict(False, True, i, 1)))
					epoch_loss += loss_v
					acc_v = sess.run(accuracy,feed_dict = feed_dict(True,j))
					acc += acc_v
				acc /= s
				epoch_loss /= s
				# train_loss = sess.run(train_loss_summary, feed_dict={train_average_loss: epoch_loss})
				# train_writer.add_summary(train_loss, step + 1)

				print("\nEpoch #%-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
					  (epoch, epoch_loss, end - start))
				print("Epoch #%-5d | Train accuracy: %.2f" % (epoch, acc))
				# test loss
				start = time.time()
				epoch_loss = 0
				acc = 0
				s = data_sets.eval_pos.shape[0]//BS
				for j in range(s):
					loss_v = sess.run(loss, feed_dict=feed_dict(False, j))
					epoch_loss += loss_v
					acc_v = sess.run(accuracy,feed_dict = feed_dict(False,j))
					acc += acc_v
				# print(epoch_loss)
				# print(FLAGS.pack_size * BS)
				acc /= s
				epoch_loss /= s
				# test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
				# train_writer.add_summary(test_loss, step + 1)
				# test_writer.add_summary(test_loss, step + 1)
				print("Epoch #%-5d | Eval  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
					  (epoch, epoch_loss, start - end))
				print("Epoch #%-5d | Eval  accuracy: %.2f" % (epoch, acc))