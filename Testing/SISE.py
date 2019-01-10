import random
import numpy
import math

from matplotlib import pyplot

from library import funs
from library.funs import parameters_as_string, print_plot, calculate_results_table

from mlp.NeuralNetwork import NeuralNetwork


def normalize(to_normalize):
    vmin = min(to_normalize)
    vmax = max(to_normalize)
    normalized = []
    for var in to_normalize:
        normalized.append((var - vmin) / (vmax - vmin))
    return normalized


def denormalize(to_normalize, normalized):
    vmin = min(to_normalize)
    vmax = max(to_normalize)
    denormalized = []
    for var in normalized:
        denormalized.append((var * (vmax - vmin)) + vmin)
    return denormalized


def task_SISE():
    epochs = 2000
    hidden_nodes = 5
    learning_rate = 0.2
    momentum = 0.9
    bias = False
    network = NeuralNetwork(input_nodes=1, hidden_nodes=hidden_nodes, output_nodes=1,
                            learning_rate=learning_rate,
                            momentum=momentum, bias=bias, epochs=epochs)

    input_list = []
    input_list = list(range(1, 100))
    output = []
    for var in input_list:
        output.append(math.sqrt(var))

    c = list(zip(input_list, output))

    # shuffle in the same way
    random.shuffle(c)

    input_list, output = zip(*c)

    input_norm = normalize(input_list)
    output_norm = normalize(output)
    for e in range(epochs):
        for i in range(len(input_norm)):
            network.train_manual_epochs([input_norm[i]], [output_norm[i]], e)  # inputs need to be arrays

    fin = []
    query_list = []
    for i in range(len(input_norm)):
        query_list.append(input_norm[i])
        fin.append(network.query([query_list[i]]))

    numpy.set_printoptions(suppress=True)  # avoid e-05 notation
    print(output)

    results = denormalize(output, fin)
    for e in range(len(results)):
        print(input_list[e], results[e])

    result_tab = numpy.zeros(shape=(len(input_norm), len(input_norm)))

    for i in range(len(fin)):
        result_tab[numpy.argmax(query_list[i])][numpy.argmax(fin[i])] += 1

    print('Results table')
    print(result_tab)
    parameters = parameters_as_string(hidden_nodes, learning_rate, momentum, epochs, bias)
    funs.print_plot(network.sampling_iteration, network.errors_for_plot, 'Error plot ' + parameters)

    flat_results = [item for sublist in results for item in sublist]
    flat_input = list(input_list)
    flat_output = list(output)
    title = "Expected(green) and calculated(red) values"
    pyplot.figure(title)
    pyplot.plot(flat_input, flat_results, 'r.')
    pyplot.plot(flat_input, flat_output, 'g.')
    pyplot.grid(axis='both', color='black', which='major', linestyle='--', linewidth=1)
    pyplot.xlabel("x")
    pyplot.ylabel("sqrt(x)")
    pyplot.suptitle(title)
    pyplot.savefig('plotValues.png')
    pyplot.show()
