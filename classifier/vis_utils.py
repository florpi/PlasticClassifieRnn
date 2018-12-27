import tensorflow as tf
import numpy as np

import textwrap
import re
import io
import itertools
import matplotlib
from matplotlib.pyplot import cm

import tfplot

# Import lazy modules
from matplotlib import figure
from matplotlib.backends import backend_agg

# From our repo
from decode_tfrecords import NUM_BANDS

class ConfusionMatrixSaverHook(tf.train.SessionRunHook):
    """
    Saves a confusion matrix as a Summary so that it can be shown in tensorboard
    """

    def __init__(self, labels, confusion_matrix_tensor_name, summary_writer):
        """Initializes a `SaveConfusionMatrixHook`.

        :param labels: Iterable of String containing the labels to print for each
                       row/column in the confusion matrix.
        :param confusion_matrix_tensor_name: The name of the tensor containing the confusion
                                             matrix
        :param summary_writer: The summary writer that will save the summary
        """
        self.confusion_matrix_tensor_name = confusion_matrix_tensor_name
        self.labels = labels
        self._summary_writer = summary_writer

    def end(self, session):
        cm = tf.get_default_graph().get_tensor_by_name(
                self.confusion_matrix_tensor_name + ':0').eval(session=session).astype(int)
        globalStep = tf.train.get_global_step().eval(session=session)
        figure = self._plot_confusion_matrix(cm)
        summary = self._figure_to_summary(figure)
        self._summary_writer.add_summary(summary, globalStep)

    def _figure_to_summary(self, fig):
        """
        Converts a matplotlib figure ``fig`` into a TensorFlow Summary object
        that can be directly fed into ``Summary.FileWriter``.
        :param fig: A ``matplotlib.figure.Figure`` object.
        :return: A TensorFlow ``Summary`` protobuf object containing the plot image
                 as a image summary.
        """

        # attach a new canvas if not exists
        if fig.canvas is None:
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        # get PNG data from the figure
        png_buffer = io.BytesIO()
        fig.canvas.print_png(png_buffer)
        png_encoded = png_buffer.getvalue()
        png_buffer.close()

        summary_image = tf.Summary.Image(height=h, width=w, colorspace=4,  # RGB-A
                                      encoded_image_string=png_encoded)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.confusion_matrix_tensor_name, image=summary_image)])
        return summary

    def _plot_confusion_matrix(self, cm, normalize=True):
        '''
        :param cm: A confusion matrix: A square ```numpy array``` of the same size as self.labels
    `   :return:  A ``matplotlib.figure.Figure`` object with a numerical and graphical representation of the cm array
        '''
        if normalize:
            cm = cm.astype(np.float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
            cm = (100.*cm).astype(np.int)
        numClasses = len(self.labels)

        fig = matplotlib.figure.Figure(figsize=(numClasses, numClasses), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cm, cmap='Oranges')

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in self.labels]
        classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted')
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(numClasses), range(numClasses)):
            ax.text(j, i, int(cm[i, j]) if cm[i, j] != 0 else '.', horizontalalignment="center", verticalalignment='center', color="black")
        fig.set_tight_layout(True)
        return fig

def plot_light_curves(*feats_values, feats_keys=None):
    """
    Make a scatter plot of the flux and its error, and a second one of the agumented_flux.
    Note that inputs to this function are no longer tensors, but numpy arrays.
    """
    feats = dict(zip(feats_keys, feats_values))
    object_id = feats['object_id']

    fig, ax = tfplot.subplots(6, 1, figsize=(12, 36))
    colors = cm.rainbow(np.linspace(0, 1, NUM_BANDS))
    # Original flux and flux_err
    for band in range(NUM_BANDS):
        n_samples = feats['band_%i/num_samples'%band]
        ax[0].errorbar(feats['band_%i/mjd'%band][:n_samples],
                       feats['band_%i/original_flux'%band][:n_samples],
                       yerr=feats['band_%i/flux_err'%band][:n_samples],
                       color=colors[band],
                       label='band %i'%band,
                       fmt='o')
    ax[0].legend()
    ax[0].set_title('(Object %i) original flux'%object_id)
    # Augmented flux
    for band in range(NUM_BANDS):
        n_samples = feats['band_%i/num_samples'%band]
        ax[1].scatter(feats['band_%i/mjd'%band][:n_samples],
                      feats['band_%i/augmented_flux'%band][:n_samples],
                      color=colors[band],
                      label='band %i'%band)
    ax[1].legend()
    ax[1].set_title('(Object %i) augmented flux'%object_id)
    # Augmented and preprocessed flux
    for band in range(NUM_BANDS):
        n_samples = feats['band_%i/num_samples'%band]
        ax[2].scatter(feats['band_%i/preprocessed_time_diff'%band][:n_samples],
                      feats['band_%i/preprocessed_flux'%band][:n_samples],
                      color=colors[band],
                      label='band %i'%band)
    ax[2].legend()
    ax[2].set_title('(Object %i) augmented and preprocessed flux'%object_id)
    # Aggregated bands
    total_samples = 0
    for band in range(NUM_BANDS):
        total_samples += feats['band_%i/num_samples'%band]
    ax[3].set_title('(Object %i) Aggregated flux features'%object_id)
    # Preprocessed DFT
    for band in range(NUM_BANDS):
        n_samples = feats['band_%i/dft/num_samples'%band]
        dft_mag = feats['band_%i/dft'%band][:n_samples, 0]
        ax[4].plot(dft_mag,
                   color=colors[band],
                   label='band %i'%band)
    ax[4].legend()
    ax[4].set_title('(Object %i) preprocessed magnitude NDFT'%object_id)

    for band in range(NUM_BANDS):
        n_samples = feats['band_%i/dft/num_samples'%band]
        dft_phase = feats['band_%i/dft'%band][:n_samples, 1]
        ax[5].plot(dft_phase,
                   color=colors[band],
                   label='band %i'%band)
    ax[5].legend()
    ax[5].set_title('(Object %i) preprocessed phase NDFT'%object_id)
    fig.subplots_adjust(0, 0, 1, 1)  # use tight layout (no margins)
    return fig

