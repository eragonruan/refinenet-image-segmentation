import tensorflow as tf
from tensorflow.contrib import slim
from nets import resnet_v1
from utils.training import get_valid_logits_and_labels
FLAGS = tf.app.flags.FLAGS

def unpool(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])


def ResidualConvUnit(inputs,features=256,kernel_size=3):
    net=tf.nn.relu(inputs)
    net=slim.conv2d(net, features, kernel_size)
    net=tf.nn.relu(net)
    net=slim.conv2d(net,features,kernel_size)
    net=tf.add(net,inputs)
    return net

def ChainedResidualPooling(inputs,features=256):
    net_relu=tf.nn.relu(inputs)
    net=slim.max_pool2d(net_relu, [5, 5],stride=1,padding='SAME')
    net=slim.conv2d(net,features,3)
    net_sum_1=tf.add(net,net_relu)

    net = slim.max_pool2d(net_relu, [5, 5], stride=1, padding='SAME')
    net = slim.conv2d(net, features, 3)
    net_sum_2=tf.add(net,net_sum_1)

    return net_sum_2


def MultiResolutionFusion(high_inputs=None,low_inputs=None,features=256):

    if high_inputs is None:#refineNet block 4
        rcu_low_1 = low_inputs[0]
        rcu_low_2 = low_inputs[1]

        rcu_low_1 = slim.conv2d(rcu_low_1, features, 3)
        rcu_low_2 = slim.conv2d(rcu_low_2, features, 3)

        return tf.add(rcu_low_1,rcu_low_2)

    else:
        rcu_low_1 = low_inputs[0]
        rcu_low_2 = low_inputs[1]

        rcu_low_1 = slim.conv2d(rcu_low_1, features, 3)
        rcu_low_2 = slim.conv2d(rcu_low_2, features, 3)

        rcu_low = tf.add(rcu_low_1,rcu_low_2)

        rcu_high_1 = high_inputs[0]
        rcu_high_2 = high_inputs[1]

        rcu_high_1 = unpool(slim.conv2d(rcu_high_1, features, 3),2)
        rcu_high_2 = unpool(slim.conv2d(rcu_high_2, features, 3),2)

        rcu_high = tf.add(rcu_high_1,rcu_high_2)

        return tf.add(rcu_low, rcu_high)


def RefineBlock(high_inputs=None,low_inputs=None):

    if high_inputs is None: # block 4
        rcu_low_1= ResidualConvUnit(low_inputs, features=256)
        rcu_low_2 = ResidualConvUnit(low_inputs, features=256)
        rcu_low = [rcu_low_1, rcu_low_2]

        fuse = MultiResolutionFusion(high_inputs=None, low_inputs=rcu_low, features=256)
        fuse_pooling = ChainedResidualPooling(fuse, features=256)
        output = ResidualConvUnit(fuse_pooling, features=256)
        return output
    else:
        rcu_low_1 = ResidualConvUnit(low_inputs, features=256)
        rcu_low_2 = ResidualConvUnit(low_inputs, features=256)
        rcu_low = [rcu_low_1, rcu_low_2]

        rcu_high_1 = ResidualConvUnit(high_inputs, features=256)
        rcu_high_2 = ResidualConvUnit(high_inputs, features=256)
        rcu_high = [rcu_high_1, rcu_high_2]

        fuse = MultiResolutionFusion(rcu_high, rcu_low,features=256)
        fuse_pooling = ChainedResidualPooling(fuse, features=256)
        output = ResidualConvUnit(fuse_pooling, features=256)
        return output



def model(images, weight_decay=1e-5, is_training=True):
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_101(images, is_training=is_training, scope='resnet_v1_101')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {'decay': 0.997,'epsilon': 1e-5,'scale': True,'is_training': is_training}
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):

            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))

            g = [None, None, None, None]
            h = [None, None, None, None]

            for i in range(4):
                h[i]=slim.conv2d(f[i], 256, 1)
            for i in range(4):
                print('Shape of h_{} {}'.format(i, h[i].shape))

            g[0]=RefineBlock(high_inputs=None,low_inputs=h[0])
            g[1]=RefineBlock(g[0],h[1])
            g[2]=RefineBlock(g[1],h[2])
            g[3]=RefineBlock(g[2],h[3])
            #g[3]=unpool(g[3],scale=4)
            F_score = slim.conv2d(g[3], 21, 1, activation_fn=tf.nn.relu, normalizer_fn=None)

    return F_score


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    images=tf.to_float(images)
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def loss(annotation_batch,upsampled_logits_batch,class_labels):
    valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(
        annotation_batch_tensor=annotation_batch,
        logits_batch_tensor=upsampled_logits_batch,
        class_labels=class_labels)

    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                              labels=valid_labels_batch_tensor)

    cross_entropy_sum = tf.reduce_mean(cross_entropies)
    tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

    return cross_entropy_sum

