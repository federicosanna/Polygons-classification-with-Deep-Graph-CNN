# Written by Federico Sanna
# 30/05/2019
# Script to generate input dataset to train the Graph dataset

# Usage Info:
# Every time that this script is run it writes into a file called
# POLY.txt a number of nodes (as specified in poly_to_generate) of
# the type either concave, convex, or both, as specified.
# For more flexibility the number of total graph needs to be specified
# by the user at the beginning of the file. This is to avoid that adding
# nodes results in corrupting the format of the file. 

import numpy as np
from numpy import array, newaxis, expand_dims
import torch
import matplotlib.pyplot as plt
import random



# generate_convex_poly = True
# generate_concave_poly = False
generate_convex_poly = False
generate_concave_poly = True
poly_to_generate = 2500

def generate_polygons(rmin, rmax, n_gons, n_out):
    """    Generates n_out polygons of n_gons sides
            from a circle with a radius between rmin and rmax """
    # Creating a list of n_out random radii with values
    # between rmin and rmax (can be redundant)
    listr = np.random.ranf(n_out) * (rmax - rmin) + rmin

    # Initializing the Matrix of angles of size (n_gons, n_out)
    mat_theta = np.zeros((n_gons, n_out))
    thetanormal = [k * 2 * np.pi / n_gons for k in range(n_gons)]

    for i in range(n_out):
        mat_theta[:, i] = [np.random.normal(thetanormal[k], listr[i] / 9) for k in range(n_gons)]

    x = listr * np.cos(mat_theta)  # Xcoordinates
    y = listr * np.sin(mat_theta)  # Ycoordinates

    return (x, y)


def generate_poly_with_variable_n_gons(n_gons, size_of_ds_poly):
    """ Generate a tuple of size size_of_ds_poly, cointaing pairs of arrays
        representing the x and y coordinates. Generates the vertices of polygons
        with n_gons corners.
        n_gons is required to be a list of integers with the numbers of corners
        of the polygons wanted to be part of the dataset.
        The dataset will be split equaly between the different types of polygons.

        Inputs:
        - n_gons (type=list): represent types of polygons that you want to
          be part of the dataset
        - size_of_ds_poly: number of polygons to be included in the dataset
        Output:
        - tuple of size 'size_of_ds_poly' cointaing elements made of two arrays
          each of size (1 x n_gons) representing the coordinates of the vertices.
        Example:
        generate_poly_with_variable_n_gons([5, 6], 3)
            a Tuple with 3 elements corresponding to:
            1 pair of arrays with the pentagon coordinates:
            [x_1,x_2,x_3,x_4,x_5]
            [y_1,y_2,y_3,y_4,y_5]

            2 pairs of arrays with the exagons coordinates:
            [x_11,x_12,x_13,x_14,x_15,x_16]
            [y_11,y_12,y_13,y_14,y_15,y_16]

            [x_21,x_22,x_23,x_24,x_25,x_26]
            [y_21,y_22,y_23,y_24,y_25,y_26]
        """
    # How many images with a certain number of corners
    images_per_n_gons = int(size_of_ds_poly / len(n_gons))
    # Initialise list for the vertices
    vertices = [None] * size_of_ds_poly
    # Since size_of_ds_poly/len(n_gons) could be a non-integer, we need to take
    # care of the last cases with 2 separate for loops
    # Fill the list with poly of different numbers of corners
    if len(n_gons) > 1:
        for i in range(len(n_gons) - 1):
            for j in range(images_per_n_gons):
                vertices[j + i * images_per_n_gons] = generate_polygons(0.9, 0.6, n_gons[i], 1)
    # Last one fills up until the end of the ds size
    for i in range(size_of_ds_poly - (images_per_n_gons * (len(n_gons) - 1))):
        vertices[(images_per_n_gons * (len(n_gons) - 1)) + i] = generate_polygons(0.9, 0.6, n_gons[len(n_gons) - 1], 1)

    return vertices

def point_to_class_index(y):
    _, c = y.view(-1).max(dim=0)
    return c

canvas_size = 64
vertices = generate_poly_with_variable_n_gons([5], poly_to_generate)
to_be_displayed = torch.zeros((canvas_size, canvas_size, 1), dtype=torch.uint8)

# Set next line to true if you want to have two float points as features for the node
# If set True, you will have as features 2 float numbers between -1 and 1. One of them
# represents the x coordinate, the second one represents the y coordinate.
# If set to false, you will have a single number identifying what pixel of the figure
# is representing the corner.
use_two_float_cord = True


if generate_convex_poly == True:
    if use_two_float_cord == False:
        for x in vertices:
            # There are going to be size_of_ds_poly iterations in the loop
            # x is going to take the value of the vertex for each iteration
            # so in every loop we need to work on x
            # From vertices in the range -1,1 to vertices in the range 1,64
            vertices_x = ((x[0] + 1) * (canvas_size / 2)).round()  # need to be carefull that they end up arriving to 64, not 63
            vertices_y = ((x[1] + 1) * (canvas_size / 2)).round()
            to_be_displayed = torch.zeros((canvas_size, canvas_size, 1), dtype=torch.uint8)
            # Print number of nodes in a Graph and label of the Graph
            with open('POLY.txt', 'a+') as f:
                f.write('5 0' + '\n')
            # Populate it with ones in correspondance of vertices
            for k in range(vertices_x.shape[0]):
                # Create 64x64 zero tensor
                base_array = torch.zeros((canvas_size, canvas_size, 1), dtype=torch.uint8)
                base_array[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1][0] = 1
                to_be_displayed[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1][0] = 1
                coordinates_feature = point_to_class_index(base_array)
                # Prepare the number to be written in the text file
                if k == 0:
                    connectivity = '1 4 '
                elif k == 1:
                    connectivity = '0 2 '
                elif k == 2:
                    connectivity = '1 3 '
                elif k == 3:
                    connectivity = '2 4 '
                elif k == 4:
                    connectivity = '0 3 '
                with open('POLY.txt', 'a+') as f:
                    f.write('0 2 ' + connectivity + str(int(coordinates_feature)) + '\n')
            # print(x, " ", end='')
    elif use_two_float_cord == True:
        for x in vertices:
            # There are going to be size_of_ds_poly iterations in the loop
            # x is going to take the value of the vertex for each iteration
            # so in every loop we need to work on x
            # From vertices in the range -1,1 to vertices in the range 1,64
            vertices_x = ((x[0] + 1) * (canvas_size / 2)).round()  # need to be carefull that they end up arriving to 64, not 63
            vertices_y = ((x[1] + 1) * (canvas_size / 2)).round()
            to_be_displayed = torch.zeros((canvas_size, canvas_size, 1), dtype=torch.uint8)
            # Print number of nodes in a Graph and label of the Graph
            with open('POLY.txt', 'a+') as f:
                f.write('5 0' + '\n')
            # Populate it with ones in correspondance of vertices
            for k in range(x[0].shape[0]):
                # Fill 64x64 tensor to display
                to_be_displayed[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1][0] = 1
                # Prepare the number to be written in the text file
                if k == 0:
                    connectivity = '1 4 '
                elif k == 1:
                    connectivity = '0 2 '
                elif k == 2:
                    connectivity = '1 3 '
                elif k == 3:
                    connectivity = '2 4 '
                elif k == 4:
                    connectivity = '0 3 '
                with open('POLY.txt', 'a+') as f:
                    f.write('0 2 ' + connectivity + "{:.6f}".format(float(x[0][k])) + ' ' + "{:.6f}".format(float(x[1][k])) + '\n')
            # print(x, " ", end='')


if generate_concave_poly == True:
    if use_two_float_cord == False:
        for x in vertices:
            # There are going to be size_of_ds_poly iterations in the loop
            # x is going to take the value of the vertex for each iteration
            # so in every loop we need to work on x

            # Here we are going to change the coordinates of one of the corners at random
            # so that the polygon becomes concave.
            # To do that we are going to pick a corner (c), select the two adjacent corners (a and b).
            # We are then going to project corner c simmetricaly with respect to b-a so that
            # then the angle in c is going to be greater than 180°.
            index_corner_to_be_moved = random.randint(0, x[0].shape[0]-1)
            index_corner_a = (index_corner_to_be_moved - 1) % x[0].shape[0]-1
            index_corner_b = (index_corner_to_be_moved + 1) % x[0].shape[0]-1
            c = np.array([x[0][index_corner_to_be_moved], x[1][index_corner_to_be_moved]])
            a = np.array([x[0][index_corner_a], x[1][index_corner_a]])
            b = np.array([x[0][index_corner_b], x[1][index_corner_b]])
            # We set how much far inside the polygon the moved corner is going to be
            coefficient = 1.3
            # Compute new position of c
            c = (((b - a) / 2) - (c - a)) * coefficient + c
            # Update position of c
            x[0][index_corner_to_be_moved] = c[0]
            x[1][index_corner_to_be_moved] = c[1]
            # From vertices in the range -1,1 to vertices in the range 1,64
            vertices_x = ((x[0] + 1) * (canvas_size / 2)).round()  # need to be carefull that they end up arriving to 64, not 63
            vertices_y = ((x[1] + 1) * (canvas_size / 2)).round()
            to_be_displayed = torch.zeros((canvas_size, canvas_size, 1), dtype=torch.uint8)
            # Print number of nodes in a Graph and label of the Graph
            with open('POLY.txt', 'a+') as f:
                f.write('5 1' + '\n')
            # Populate it with ones in correspondance of vertices
            for k in range(vertices_x.shape[0]):
                # Create 64x64 zero tensor
                base_array = torch.zeros((canvas_size, canvas_size, 1), dtype=torch.uint8)
                base_array[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1][0] = 1
                to_be_displayed[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1][0] = 1
                coordinates_feature = point_to_class_index(base_array)
                # Prepare the number to be written in the text file
                if k == 0:
                    connectivity = '1 4 '
                elif k == 1:
                    connectivity = '0 2 '
                elif k == 2:
                    connectivity = '1 3 '
                elif k == 3:
                    connectivity = '2 4 '
                elif k == 4:
                    connectivity = '0 3 '
                with open('POLY.txt', 'a+') as f:
                    f.write('0 2 ' + connectivity + str(int(coordinates_feature)) + '\n')
            # print(x, " ", end='')
    elif use_two_float_cord == True:
        for x in vertices:
            # There are going to be size_of_ds_poly iterations in the loop
            # x is going to take the value of the vertex for each iteration
            # so in every loop we need to work on x

            # Here we are going to change the coordinates of one of the corners at random
            # so that the polygon becomes concave.
            # To do that we are going to pick a corner (c), select the two adjacent corners (a and b).
            # We are then going to project corner c simmetricaly with respect to b-a so that
            # then the angle in c is going to be greater than 180°.
            index_corner_to_be_moved = random.randint(0, x[0].shape[0] - 1)
            index_corner_a = (index_corner_to_be_moved - 1) % x[0].shape[0] - 1
            index_corner_b = (index_corner_to_be_moved + 1) % x[0].shape[0] - 1
            c = np.array([x[0][index_corner_to_be_moved], x[1][index_corner_to_be_moved]])
            a = np.array([x[0][index_corner_a], x[1][index_corner_a]])
            b = np.array([x[0][index_corner_b], x[1][index_corner_b]])
            # We set how much far inside the polygon the moved corner is going to be
            coefficient = 1.3
            # Compute new position of c
            c = (((b - a) / 2) - (c - a)) * coefficient + c
            # Update position of c
            x[0][index_corner_to_be_moved] = c[0]
            x[1][index_corner_to_be_moved] = c[1]

            # From vertices in the range -1,1 to vertices in the range 1,64
            vertices_x = ((x[0] + 1) * (canvas_size / 2)).round()  # need to be carefull that they end up arriving to 64, not 63
            vertices_y = ((x[1] + 1) * (canvas_size / 2)).round()
            to_be_displayed = torch.zeros((canvas_size, canvas_size, 1), dtype=torch.uint8)
            # Print number of nodes in a Graph and label of the Graph
            with open('POLY.txt', 'a+') as f:
                f.write('5 1' + '\n')
            # Populate it with ones in correspondance of vertices
            for k in range(x[0].shape[0]):
                # Fill 64x64 tensor to display
                to_be_displayed[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1][0] = 1
                # Prepare the number to be written in the text file
                if k == 0:
                    connectivity = '1 4 '
                elif k == 1:
                    connectivity = '0 2 '
                elif k == 2:
                    connectivity = '1 3 '
                elif k == 3:
                    connectivity = '2 4 '
                elif k == 4:
                    connectivity = '0 3 '
                with open('POLY.txt', 'a+') as f:
                    f.write('0 2 ' + connectivity + "{:.6f}".format(float(x[0][k])) + ' ' + "{:.6f}".format(float(x[1][k])) + '\n')
            # print(x, " ", end='')

to_be_displayed1 = to_be_displayed.numpy()
to_be_displayed1 = to_be_displayed1.reshape(64,64)
print(to_be_displayed1.size)
plt.imshow(to_be_displayed1, cmap='gray')
plt.show()
