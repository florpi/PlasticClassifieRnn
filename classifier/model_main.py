from absl import flags
import tensorflow as tf

# From our repo
import  model_lib

flags.DEFINE_string('model', None, 'model to use (for now FCN or RNN)')
flags.DEFINE_string('model_dir', None, 'Path to output model directory '
                    'where event and checkpoint files will be written.')
flags.DEFINE_string('dataset_dir', None, 'Path to the data directory.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_epochs', None, 'Number of train steps.')
flags.DEFINE_integer('validation_fold', 1, 'Fold number to perform validation on.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')

FLAGS = flags.FLAGS

def main(unused_argv):
    """
    """
    flags.mark_flag_as_required('model')
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('dataset_dir')
    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                    save_summary_steps=500, #100,
                                    save_checkpoints_steps=500, #200,
                                    keep_checkpoint_max=100,
                                    #keep_checkpoint_every_n_hours=1,
                                    log_step_count_steps=250) #50)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(model=FLAGS.model,
                                                                run_config=config,
                                                                dataset_dir=FLAGS.dataset_dir,
                                                                batch_size=FLAGS.batch_size,
                                                                num_epochs=FLAGS.num_epochs,
                                                                validation_fold=FLAGS.validation_fold,
                                                                learning_rate=FLAGS.learning_rate)

    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fn = train_and_eval_dict['eval_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    train_spec, eval_spec = model_lib.create_train_and_eval_specs(train_input_fn,
                                                                  eval_input_fn,
                                                                  predict_input_fn,
                                                                  train_steps)
    # Train and envaluation loops
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    # either DEBUG, INFO, WARN, ERROR, or FATAL
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
