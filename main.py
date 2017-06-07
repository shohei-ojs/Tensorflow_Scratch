import os
import input_data
import numpy as np
import tensorflow as tf




tf.app.flags.DEFINE_string('data_batch','data_batch_1',"""使用データバッチ""")
tf.app.flags.DEFINE_integer('width_size',32,"""横""")
tf.app.flags.DEFINE_integer('height_size',32,"""高さ""")
tf.app.flags.DEFINE_integer('depth_size',3,"""チャンネル数""")
tf.app.flags.DEFINE_integer('max_step', 10000,"""訓練ステップ数""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint/', """チェックポイント保存先""")
tf.app.flags.DEFINE_string('log_dir', 'logs/', """log保存先""")
tf.app.flags.DEFINE_boolean('predict', False, """推論モードフラグ""")
tf.app.flags.DEFINE_string('image_path', '', """推論する画像パス""")

FLAGS = tf.app.flags.FLAGS

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'flog', 'horse', 'ship', 'truck']


def predict():
    x = tf.placeholder(tf.float32, [None, FLAGS.width_size*FLAGS.height_size*FLAGS.depth_size])
    with tf.Session() as sess:
        output = convolution(x)
        saver = tf.train.Saver()
        load_checkpoint(sess, saver)
        image = []
        image.append(input_data.read_image(FLAGS.image_path))
        logits = sess.run(output, feed_dict={x: image})
        _class = np.argmax(logits)
        print('The category of this image is %s.' % CLASSES[_class])


def train():
    with tf.Graph().as_default():
        tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
        data = input_data.InputData()

        x = tf.placeholder(tf.float32, [None, FLAGS.width_size*FLAGS.height_size*FLAGS.depth_size])
        y = tf.placeholder(tf.float32, [None, 10])

        with tf.Session() as sess:
            output = convolution(x)
            loss_op = loss(output, y)
            train_op = train(loss_op)
            accuracy_op = accuracy(output, y)
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            for i, vals in enumerate(tf.trainable_variables()):
                tf.summary.histogram(vals.name, vals)
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()
            load_checkpoint(sess, saver)
            for i in range(FLAGS.max_step):
                labels, images = data.next_batch()
                _ = sess.run((train_op), feed_dict={x: images, y: labels}) 
                if i % 10 == 0:
                    _loss, _accuracy = sess.run((loss_op, accuracy_op), feed_dict={x: images, y: labels}) 
                    print('global step: %04d, train loss: %01.7f, train accuracy %01.5f' % (i, _loss, _accuracy))
                if i % 100 == 0 or i == FLAGS.max_step - 1:
                    summary_str = sess.run(summary_op, feed_dict={x: images, y: labels})
                    summary_writer.add_summary(summary_str, i)
                if i % 1000 == 0 or i == FLAGS.max_step - 1:
                    saver.save(sess, FLAGS.checkpoint_dir, global_step=i)
                    test_labels, test_images = data.test_data()
                    _accuracy = sess.run(accuracy_op, feed_dict={x: test_images, y: test_labels})
                    print('Test accuracy: %s' % _accuracy)
    

def convolution(images):
    batch_size = tf.shape(images)[0]
    images = tf.reshape(images, [batch_size, FLAGS.width_size, FLAGS.height_size, FLAGS.depth_size])

    output = tf.layers.conv2d(images, filters=16, kernel_size=[5, 5], strides=[2, 2], padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output, pool_size=[3, 3], strides=[2,2], padding='SAME')

    output = tf.layers.conv2d(images, filters=32, kernel_size=[5, 5], strides=[2, 2], padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output, pool_size=[3, 3], strides=[2,2], padding='SAME')

    output = tf.layers.conv2d(images, filters=64, kernel_size=[5, 5], strides=[2, 2], padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output, pool_size=[3, 3], strides=[2,2], padding='SAME')

    output = tf.contrib.layers.flatten(output)

    output = tf.layers.dense(output, 1024)
    output = tf.nn.relu(output)

    output = tf.layers.dense(output, 256)
    output = tf.nn.relu(output)

    output = tf.layers.dense(output, 10)
    output = tf.nn.softmax(output)
    return output


def loss(logits, labels):
    loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits + 1e-10), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
    return loss

def train(loss):
    return tf.train.AdamOptimizer().minimize(loss)


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


def load_checkpoint(sess, saver):
    if os.path.exists(FLAGS.checkpoint_dir + 'checkpoint'):
        print('restore parameters...')
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    else:
        print('initirize parameters...')
        init_op = tf.global_variables_initializer()
        sess.run(init_op)


def main():
    if FLAGS.predict:
        predict()
    else:
        train()


if __name__ == '__main__':
    main()