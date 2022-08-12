import sys
from csv import reader
import pygame
import math
import numpy as np
import matplotlib.pyplot as plt
# import random

hexColor = pygame.Color('#61A4BC')
title_color = pygame.Color('black')
rows = 9
MATRIX_SIZE = 9
MIDDLE_ROW_IDX = 4
MIN_ROW_SIZE = 5
EPOCHS = 10
radius = 20
colors = ['#91BAD6', '#73A5C6', '#528AAE', '#2E5984', '#1E3F66']


# parse the file received as an argument
def get_input(filePath):
    csvRows = []
    # read CSV file
    with open(filePath) as inputFile:
        csv_reader = reader(inputFile)
        header = next(csv_reader)
        # read each row
        for row in csv_reader:
            title = row[0]
            del row[0]
            csvRows.append([title, [int(val) for val in row]])
    return header, csvRows


# euclidean distance
def euclidean_dist(x1, x2):
    return np.linalg.norm(x1-x2)


# The function receives an input vector and returns the index of the vector closest to it in som,
# and the distance between them
def get_closest_idx_vector(input_vector, matrix):
    min_val = float('inf')
    min_x, min_y = 0, 0
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            if matrix[i][j] is not None:
                dist = euclidean_dist(input_vector, matrix[i][j])
                # update closest details
                if dist < min_val:
                    min_val = dist
                    min_x, min_y = i, j
    return min_x, min_y, min_val


# initialize random vector, the vector size is the size of the input vectors
def initialize_random_vector(vector_len):
    random_range = 15000
    vector = np.random.randint(0, random_range, vector_len)
    return vector


# Initialize a matrix with 61 random vectors depending on the hexagonal shape,
# so that the other cells in the matrix have none
def initialize_matrix(vector_len):
    matrix = [[None for _ in range(MATRIX_SIZE)] for _ in range(MATRIX_SIZE)]
    # The top of the hexagon
    for i in range(MIDDLE_ROW_IDX + 1):
        for j in range(MIN_ROW_SIZE + i):
            matrix[i][j] = initialize_random_vector(vector_len)
    iteration = 1
    # The bottom of the hexagon
    for i in range(MIDDLE_ROW_IDX + 1, MATRIX_SIZE):
        for j in range(MATRIX_SIZE - iteration):
            matrix[i][j] = initialize_random_vector(vector_len)
        iteration += 1
    return matrix


# The function gets an index of hexagon and returns the indexes of the neighbors
def get_neighbors(i, j, matrix):
    neighbors = []
    # invalid index
    if not (0 <= i < MATRIX_SIZE) or not (0 <= j < MATRIX_SIZE) or matrix[i][j] is None:
        return neighbors
    # left neighbor
    if j + 1 < MATRIX_SIZE and matrix[i][j + 1] is not None:
        neighbors.append((i, j + 1))
    # right neighbor
    if j - 1 >= 0:
        neighbors.append((i, j - 1))

    # The top of the hexagon
    if i < MIDDLE_ROW_IDX:
        # top left neighbor
        if i - 1 >= 0 and j - 1 >= 0:
            neighbors.append((i - 1, j - 1))
        # top right neighbor
        if i - 1 >= 0 and matrix[i - 1][j] is not None:
            neighbors.append((i - 1, j))
        # bottom left neighbor
        if i + 1 < MATRIX_SIZE and matrix[i + 1][j] is not None:
            neighbors.append((i + 1, j))
        # bottom right neighbor
        if i + 1 < MATRIX_SIZE and j + 1 < MATRIX_SIZE and matrix[i + 1][j + 1] is not None:
            neighbors.append((i + 1, j + 1))

    # The middle row of the hexagon
    elif i == MIDDLE_ROW_IDX:
        # top left neighbor
        if i - 1 >= 0 and j - 1 >= 0:
            neighbors.append((i - 1, j - 1))
        # top right neighbor
        if i - 1 >= 0 and matrix[i - 1][j] is not None:
            neighbors.append((i - 1, j))
        # bottom left neighbor
        if i + 1 < MATRIX_SIZE and j - 1 >= 0 and matrix[i + 1][j - 1] is not None:
            neighbors.append((i + 1, j - 1))
        # bottom right neighbor
        if i + 1 < MATRIX_SIZE and matrix[i + 1][j] is not None:
            neighbors.append((i + 1, j))

    # The bottom of the hexagon
    else:
        # top left neighbor
        if i - 1 >= 0:
            neighbors.append((i - 1, j))
        # top right neighbor
        if i - 1 >= 0 and j + 1 < MATRIX_SIZE and matrix[i - 1][j + 1] is not None:
            neighbors.append((i - 1, j + 1))
        # bottom left neighbor
        if i + 1 < MATRIX_SIZE and j - 1 >= 0 and matrix[i + 1][j - 1] is not None:
            neighbors.append((i + 1, j - 1))
        # bottom right neighbor
        if i + 1 < MATRIX_SIZE and matrix[i + 1][j] is not None:
            neighbors.append((i + 1, j))
    return neighbors


# The function draws an regular polygon for drawing hexagons
def draw_regular_polygon(surface, color, vertex_count, radius, position, width=0):
    n, r = vertex_count, radius
    x, y = position
    pygame.draw.polygon(surface, color, [
        (x + r * math.cos(math.pi / 2 + 2 * math.pi * i / n), y + r * math.sin(math.pi / 2 + 2 * math.pi * i / n))
        for i in range(n)
    ], width)
    pygame.display.update()


# show hexagon map on screen
def show_screen(map_input_vectors, input_vectors):
    pygame.init()
    screen = pygame.display.set_mode([750, 500])
    screen.fill('white')
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        graphic(map_input_vectors, input_vectors, screen)


# Hexagonal drawing in a particular index received as input
def draw_hexagon_by_index(i, j, color, screen):
    y, x_start = 0, 0
    # init location by index
    if i == 0:
        x_start, y = 300, 100
    elif i == 1:
        x_start, y = 280, 130
    elif i == 2:
        x_start, y = 260, 160
    elif i == 3:
        x_start, y = 240, 190
    elif i == 4:
        x_start, y = 220, 220
    elif i == 5:
        x_start, y = 240, 250
    elif i == 6:
        x_start, y = 260, 280
    elif i == 7:
        x_start, y = 280, 310
    elif i == 8:
        x_start, y = 300, 340
    # draw hexagon
    draw_regular_polygon(screen, color, 6, radius, (x_start + j * 40, y))


# The function returns the socioeconomic average of vectors mapped to hexagon
def get_avg(indexes, input_vectors):
    # if the list is empty, returns -1 (to paint the appropriate hexagon in black)
    if len(indexes) == 0:
        return -1
    sum_economy = 0
    # calculate the average
    for index in indexes:
        sum_economy += input_vectors[index][0]
    return sum_economy / len(indexes)


# The function returns the hexagonal color corresponding to the socioeconomic mean of hexagonal vectors
def get_color(avg):
    if avg == -1:
        return pygame.Color('black')
    if 0 <= avg <= 2:
        return colors[0]
    elif 2 < avg <= 4:
        return colors[1]
    elif 4 < avg <= 6:
        return colors[2]
    elif 6 < avg <= 8:
        return colors[3]
    elif 8 < avg <= 10:
        return colors[4]


# Drawing 61 hexagons as map
def graphic(map_input_vectors, input_vectors, screen):
    num_of_hexagon = 5
    add = True
    for i in range(rows):
        for j in range(num_of_hexagon):
            indexes = map_input_vectors[i][j]
            avg = get_avg(indexes, input_vectors)
            draw_hexagon_by_index(i, j, get_color(avg), screen)
        # determine the number of hexagons in each row
        if num_of_hexagon == 8 and add:
            num_of_hexagon += 1
            add = False
        elif add:
            num_of_hexagon += 1
        else:
            num_of_hexagon -= 1


# Update the som vector to bring it closer to the input vector
def update_vector(input_vector, som_vector, update_percentage):
    # go over every cell and cell and update the value according to the distance
    for i in range(len(input_vector)):
        if input_vector[i] > som_vector[i]:
            dist = input_vector[i] - som_vector[i]
            som_vector[i] += update_percentage * dist
        else:
            dist = som_vector[i] - input_vector[i]
            som_vector[i] -= update_percentage * dist
    return som_vector


# The som algorithm
def som_algorithm(input_vectors, matrix):
    for i in range(EPOCHS):
        for vector in input_vectors:
            # Getting the closet vector
            x, y, dist = get_closest_idx_vector(vector, matrix)
            updated_indexes = []
            # Update the closest vector with an update percentage of 0.8
            update_percentage = 0.8
            updated_vector = update_vector(vector, matrix[x][y], update_percentage)
            matrix[x][y] = updated_vector
            updated_indexes.append((x, y))
            # Getting the neighbors of the closest vector
            neighbors = get_neighbors(x, y, matrix)
            # Update the neighbors of the closest vector with an update percentage of 0.3
            update_percentage = 0.3
            neighbors_deg2 = []
            for n in neighbors:
                if n not in updated_indexes:
                    updated_indexes.append(n)
                    updated_vector = update_vector(vector, matrix[n[0]][n[1]], update_percentage)
                    matrix[n[0]][n[1]] = updated_vector
                    # save the neighbors of the neighbors in a list
                    neighbors_deg2.append(get_neighbors(n[0], n[1], matrix))
            flat_list_neighbors = [x for xs in neighbors_deg2 for x in xs]
            # Update the neighbors of the neighbors of the closest vector with an update percentage of 0.1
            update_percentage = 0.1
            for n in flat_list_neighbors:
                if n not in updated_indexes:
                    updated_indexes.append(n)
                    updated_vector = update_vector(vector, matrix[n[0]][n[1]], update_percentage)
                    matrix[n[0]][n[1]] = updated_vector

    # Creating a map so that in each cell we save the indexes of the vectors
    # that closest to the appropriate vector in the hexagon grid
    sum_distances = 0
    map_input_vectors = [[[] for _ in range(MATRIX_SIZE)] for _ in range(MATRIX_SIZE)]
    for i in range(len(input_vectors)):
        x, y, dist = get_closest_idx_vector(input_vectors[i], matrix)
        map_input_vectors[x][y].append(i)
        sum_distances += dist
    return map_input_vectors, sum_distances / len(input_vectors)


# Get a list of cities mapped to index i, j
def get_cities(i, j, titles, map_input_vectors):
    indexes = map_input_vectors[i][j]
    cities = []
    for index in indexes:
        cities.append(titles[index])
    return cities


# Run the algorithm several times and choose the best solution
def multiple_runs_som(vector_length, input_vectors):
    iterations_num = 10
    best_map = []
    best_matrix = []
    min_dist = float('inf')
    distances = []
    for i in range(iterations_num):
        # run som algorithm with a new random grid
        mtx = initialize_matrix(vector_length)
        map_input_vectors, avg_dist = som_algorithm(input_vectors, mtx)
        distances.append(avg_dist)
        # save best solution
        if avg_dist < min_dist:
            min_dist = avg_dist
            best_matrix = mtx
            best_map = map_input_vectors
    return best_map, best_matrix, distances


# add labels to bar chart
def add_value_label(x_list, y_list):
    for i in range(len(x_list)):
        plt.text(i, y_list[i].round(2), y_list[i].round(2), ha="center")


# Displays a graph for the averages of the distances in the various solutions
def distances_graph(distances):
    iterations = list(range(len(distances)))
    values = list(distances)
    plt.figure(figsize=(10, 5))
    # creating the bar plot
    plt.bar(iterations, values, color='maroon', width=0.4)
    add_value_label(iterations, values)
    plt.xlabel("Solution number")
    plt.ylabel("Average distances")
    plt.title("Average distances in each solution")
    plt.show()


def main():
    header, csvRows = get_input(sys.argv[1])
    vector_length = len(csvRows[0][1])
    # Mixing the input rows -- test for Section C in the exercise
    # random.shuffle(csvRows)
    # Sorting the rows by the economic index -- test for Section C in the exercise
    # csvRows = sorted(csvRows, key=lambda x: x[1][0])

    input_vectors = []
    titles = []
    # save the input vectors and the titles in arrays
    for row in csvRows:
        input_vectors.append(row[1])
        titles.append(row[0])
    # run the algorithm 10 times and get the best solution
    map_input_vectors, mtx, distances_for_graph = multiple_runs_som(vector_length, input_vectors)
    # display graph
    distances_graph(distances_for_graph)
    # show hexagon map
    show_screen(map_input_vectors, input_vectors)
    # print the cities mapped to each hexagon
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            if mtx[i][j] is not None:
                print(i, j, " : ", get_cities(i, j, titles, map_input_vectors))


if __name__ == '__main__':
    main()
