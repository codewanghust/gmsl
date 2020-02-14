import json
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import OrderedDict

# define hyper-parameters 
args = {
    # model configure
    'lr': 1e-4,
    'optmizer': tf.train.AdamOptimizer,
    'n_layers' : 3,
    'num_hidden_units' : [128, 64, 32],
    'act' : tf.nn.relu,
    'epochs': 1,
    'batch_size': 8,  # do not set this parameter too big if you runing this script using CPU.
    'dropout': 0.3,

    # data configure
    'train_data': './data/train.csv',
    'test_data': './data/test.csv',
    'feature_conf': './configure/features.json',
    'feature_names': ['unique_id', 'item_id', 'shop_id', 'cate_id', 'brand_id', 'qid', 'click', 'pay'],
    'feature_defaults': ['0', '0', '0', '0', '0', '0', '0', '0']
}


# load feature configure
args['fc'] = json.load(open(args['feature_conf'], 'r'))['features']


# generete feature columns
def generate_tf_columns():
    columns = OrderedDict()
    for fc in args['fc']:
        id_feature = layers.sparse_column_with_hash_bucket(
                            column_name=fc['feature_name'],
                            hash_bucket_size=fc['hash_bucket_size'])

                    
        embedding = layers.embedding_column(
                        id_feature,
                        dimension=fc["embedding_dimension"])
        columns[fc['feature_name']] = embedding
    return columns

# build dataset reader
def input_fn(data_file, num_epochs, shuffle, batch_size):
    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=args['feature_defaults'])
        features = dict(zip(args['feature_names'], columns))
        
        unique_id = features.pop('unique_id')
        click = features.pop('click')
        pay = features.pop('pay')

        click = tf.cast(tf.equal(click, '1'), tf.float32)
        pay = tf.cast(tf.equal(pay, '1'), tf.float32)

        return features, click, pay

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size*10)

    dataset = dataset.map(parse_csv)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, click, pay = iterator.get_next()
    return features, click, pay



def build_gmsl(features, click, pay, is_training):
    def __build_sub_network():
        net = features
        for inx in range(args['n_layers']):
            with tf.variable_scope('hidden_{}'.format(str(inx))):
                net = layers.fully_connected(
                                net,
                                args['num_hidden_units'][inx],
                                activation_fn=args['act'],
                                normalizer_fn=layers.batch_norm,
                                normalizer_params={'scale': True, 'is_training': is_training}
                            )

                net = tf.layers.dropout(
                                net,
                                rate=args['dropout'],
                                training=is_training,
                                name='dropout')
        return net

    
    def __build_logits(features, out_dim, name):
        with tf.variable_scope(name):
            logits = layers.fully_connected(
                                    features,
                                    out_dim,
                                    activation_fn=None
                                )
        return logits


    def __safe_log(x):
        return tf.log(tf.clip_by_value(x, 1e-8, 1.0))


    def __bce(y_true, y_pred, mask=None):
        pt_1 = y_true     * __safe_log(y_pred)
        pt_0 = (1-y_true) * __safe_log(1-y_pred)
        
        if mask is not None:
            return -tf.reduce_sum((pt_0 + pt_1)*mask) / (tf.reduce_sum(mask) + 1e-8)

        return -tf.reduce_mean(pt_0 + pt_1)



    # build ctr/cvr/gmv/... sub-networks
    with tf.variable_scope('CTR'):
        ctr_net = __build_sub_network()

    with tf.variable_scope('CVR'):
        cvr_net = __build_sub_network()

    #with tf.variable_scope('GMV'):
    #    gmv_net = __build_sub_network()


    # build sequence uints
    with tf.variable_scope('GRU'):
        cell = tf.nn.rnn_cell.GRUCell(num_units=32)
        h0 = ctr_net
        ctcvr_net, h1 = cell(cvr_net, h0)
        #pvgmv_net,_  = cell(gmv_net, h1)


    # build logits
    ctr_logits    = __build_logits(ctr_net,   1, 'ctr_output')
    cvr_logits    = __build_logits(cvr_net,   1, 'cvr_output')
    ctcvr_logits  = __build_logits(ctcvr_net, 1, 'ctcvrr_output')
    #gmv_logits   = __build_logits(gmv_net,   1, 'gmv_output')
    #pvgmv_logits = __build_logits(pvgmv_net, 1, 'pvgmv_output')


    # build prediction/losses
    ctr_prediction   = tf.nn.sigmoid(ctr_logits)
    cvr_prediction   = tf.nn.sigmoid(cvr_logits)
    ctcvr_prediction = tf.nn.sigmoid(ctcvr_logits)

    ctr_loss    = __bce(click, ctr_prediction)
    cvr_loss    = __bce(pay,   cvr_prediction, mask=click)
    ctcvrr_loss = __bce(pay  , ctcvr_prediction)


    return ctr_prediction, ctr_prediction, ctcvr_prediction, ctr_loss, cvr_loss, ctcvrr_loss
    

def run():
    # build  mode
    features, click, pay = input_fn(data_file=args['train_data'], 
                                    num_epochs=args['epochs'], 
                                    shuffle=True,
                                    batch_size=args['batch_size'])

    # build columns
    columns = generate_tf_columns()

    # transform features: string -> hash -> embedding
    features_embedding = layers.input_from_feature_columns(features, columns.values())

    p_ctr, p_cvr, p_ctcvr, ctr_loss, cvr_loss, ctcvrr_loss = \
                            build_gmsl(features_embedding, click, pay, is_training=True)

    
    # build optimizers
    loss = ctr_loss + cvr_loss + ctcvrr_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=args['lr'])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)


    op  = [train_op, p_ctr, p_cvr, p_ctcvr, ctr_loss, cvr_loss, ctcvrr_loss]

    with tf.Session() as sess:
        batch_count = 0
        sess.run(tf.global_variables_initializer())

        try:
            while True:
                values = sess.run(op)
                if batch_count % 10 == 0:
                    print('[Batch-%d] : ctr_loss=%.4f, cvr_loss=%.4f, ctcvrr_loss=%.4f' % (
                            batch_count, values[4], values[5], values[6]
                    ))
                
                if batch_count == 100:
                    return 
                
                batch_count += 1
        except tf.errors.OutOfRangeError:
            print("done.")

  
            
if __name__ == '__main__':
    run('')





