dataset = [(20, 5), (10, 8), (15, 25), (5, 2), (12, 32)]
centroid_1 = (20, 5)
centroid_2 = (10, 8)
iteration = 5

for i in range(iteration):
    cluster_1 = []
    cluster_2 = []
    print('\nIteration: ' + str(i+1))
    print('Centroid 1 = ' + str(centroid_1))
    print('Centroid 2 = ' + str(centroid_2))
    print('Feature1   Feature2   Distance to Centroid 1   Distance to ' + \
      'Centroid 2   Cluster')
    for sample in dataset:
        dist_1 = pow(pow(centroid_1[0] - sample[0], 2) + \
                     pow(centroid_1[1] - sample[1], 2), 0.5)
        dist_2 = pow(pow(centroid_2[0] - sample[0], 2) + \
                     pow(centroid_2[1] - sample[1], 2), 0.5)
        if min(dist_1, dist_2) == dist_1:
            cluster = 1
            cluster_1.append(sample)
        else:
            cluster = 2
            cluster_2.append(sample)
        print(str(sample[0]) + ' '*10 + str(sample[1]) + ' '*10 + \
              "{:.9f}".format(dist_1) + ' '*14 + "{:.9f}".format(dist_2) + \
                  ' '*14 + str(cluster))
    x_sum = 0
    y_sum = 0 
    for sample in cluster_1:
        x_sum += sample[0]
        y_sum += 0 + sample[1]
    try:
        centroid_1 = (x_sum/len(cluster_1), y_sum/len(cluster_1))
    except:
        print('Centroid 1 is too far off')
        break
    
    x_sum = 0
    y_sum = 0 
    for sample in cluster_2:
        x_sum += 0 + sample[0]
        y_sum += 0 + sample[1]
    try:
        centroid_2 = (x_sum/len(cluster_2), y_sum/len(cluster_2))
    except:
        print('Centroid 2 is too far off')
        break