import pandas
import networkx
import metis
import matplotlib.pyplot
import numpy
from itertools import combinations as all_combs

def calc_i_i(graph, cluster):#calculates the internal interconnectivity using the formula found, this one is used for a cluster
    weights_from_bisected = find_separation_weights(graph, cluster)
    ICsum = numpy.sum(weights_from_bisected)
    return ICsum

def calc_i_c(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = cluster.edges()
    weights = find_w(cluster, edges)
    return numpy.sum(weights)

#graph, cluster_i, cluser_j
def calculate_similarity_merged(g, c_i, c_j, alpha):
    edges = assign_edges(c_i, c_j, g)

    #calcualting R_I
    weights_all = find_w(g, edges)
    w_sum = numpy.sum(weights_all)
    c_i_i_i = calc_i_i(g, c_i)
    c_j_i_i = calc_i_i(g, c_j)
    R_I = w_sum / ((c_i_i_i + c_j_i_i) / 2.0)

    #calculating R_C
    if len(edges) == 0: return 0

    c_i_i_c = calc_i_c(g, c_i)
    c_j_i_c = calc_i_c(g, c_j)
    sec = numpy.mean(find_w(g, edges))
    sec_c_i = numpy.mean(find_separation_weights(g, c_i))
    sec_c_j =  numpy.mean(find_separation_weights(g, c_j))
    R_C = sec / ((c_i_i_c / (c_i_i_c + c_j_i_c) * sec_c_i) + (c_j_i_c / (c_i_i_c + c_j_i_c) * sec_c_j))

    ###
    R_C_with_alpha = numpy.power(R_C, alpha)
    similarity = R_I*R_C_with_alpha
    return similarity

def combine_most_similar(graph, clusters, a, K):
    c_i = 0
    c_j = 0
    max = 0

    for combination in all_combs(clusters, 2):
        i = combination[0]
        j = combination[1]
        if j != i:
            #here C_i and C_j checking
            graph_part_i = get_cluster(graph, i)
            graph_part_j = get_cluster(graph, j)
            # merging score of 2 clusters
            merged_res = calculate_similarity_merged(graph, graph_part_i, graph_part_j, a)

            if merged_res > max:
                #finding max_score cluster
                max = merged_res
                c_i = i
                c_j = j

    return max > 0, c_i, c_j


def CHAMELEON_START(K, K_N, M, A, dataset_pandas):
    graph = K_NEAREST_GRAPH(K_N, dataset_pandas)
    graph = prepare_for_partition(graph, M, dataset_pandas)
    for i in range(M - K):
        # check if the MINSIZE condition is satisfied from set of clusters without duplicates
        clusters = numpy.unique(dataset_pandas['cluster'])
        if len(clusters) > K:
            res = combine_most_similar(graph, clusters, A, K)

            if res[0] > 0:
                # if score is maximized merging these two clusters
                dataset_pandas.loc[dataset_pandas['cluster'] == res[2], 'cluster'] = res[1]
                for i, p in enumerate(graph.nodes()):
                    if graph.nodes[p]['cluster'] == res[2]:
                        graph.nodes[p]['cluster'] = res[1]

    return dataset_pandas


def ed(X, Y):
    point1 = numpy.array(X)
    point2 = numpy.array(Y)
    sum = numpy.sum(numpy.square(point1 - point2))
    return numpy.sqrt(sum)

def K_NEAREST_GRAPH(K_N, dataset_pandas):
    #get tuple point_arr
    #iterates over datset's rows
    #geting rows as an array now
    rows = dataset_pandas.itertuples()
    point_arr = [points[1:] for points in rows]

    #make a Knn graph and add all the instances from df to it
    #using networksx library for graphs

    #add vertices
    k_n_graph = networkx.Graph()
    n_points = len(point_arr)
    for i in range(n_points):
        k_n_graph.add_node(i)

    #add edges with respect to their weights ( weights are euclidian distance here )
    for index, point in enumerate(point_arr):
        distances = list(map(lambda x: ed(point, x), point_arr))
        # Kth closest second clust
        #indirectly sort the distances, we can later use the indexes!
        sorted_indices = numpy.argsort(distances)[1:K_N+1]
        #edges of each point!
        for i in sorted_indices:
            weight_to_add = 1/distances[i]
            k_n_graph.add_edge(index, i, weight = weight_to_add)#edges from smallest to highest distance since we indirectly sorted it
        #add the node now
        k_n_graph.nodes[index]['sth'] = point
    #k_n_graph.graph['edge_weight_attr'] = 'similarity'
    return k_n_graph


def part_graph(graph):
    #part_graph performs graph partitioning using K-way / recursive methods
    #part_graph returns multiple parameters, I am taking only one of them
    parts = metis.part_graph(graph, 2, objtype='cut')
    parts = parts[1]

    points = list(graph.nodes())

    #alter the graph here
    #print(len(points))
    for i in range(len(points)):
        graph.nodes[points[i]]['cluster'] = parts[i]
    return graph


def prepare_for_partition(graph, K, dataset):

    #initialize all graph nodes with 0 before finding max
    points = list(graph.nodes())
    for i in range(len(points)):
        graph.nodes[points[i]]['cluster'] = 0

    count_each = {}
    #assume max
    count_each[0] = len(graph.nodes())

    for i_clust in range(K-1):
        max_index = -1
        max = 0

        for i in range(len(count_each)):
            if count_each[i] > max:
                max = count_each[i]
                max_index = i
        max_nodes = []
        for n in graph.nodes:
            if graph.nodes[n]['cluster'] == max_index:
                max_nodes.append(n)

        #print("S_NODES")
        #print(s_nodes)
        # exit()
        max_graph = graph.subgraph(max_nodes)
        graph_parts = metis.part_graph(max_graph, 2, objtype='cut', ufactor=100)
        graph_parts = graph_parts[1]

        nodes = list(max_graph.nodes())

        max_graph_count = 0
        for i in range(len(nodes)):
            #add when node is 1
            if graph_parts[i] == 1:
                graph.nodes[nodes[i]]['cluster'] = i_clust + 1
                max_graph_count = max_graph_count+1

        count_each[max_index] = count_each[max_index] - max_graph_count
        count_each[i_clust + 1] = max_graph_count

    dataset['cluster'] = networkx.get_node_attributes(graph, 'cluster').values()
    return graph


def get_cluster(g, cluster_all):
    nodes = []
    for n in g.nodes:
        if g.nodes[n]['cluster'] == cluster_all:
            nodes.append(n)
    return nodes


def assign_edges(graph_part_i, graph_part_j, g):

    bisectorcut = []
    for i in graph_part_i:
        for j in graph_part_j:
            if i in g:#if a is in the graph
                if j in g[i]:#if b is g[a]
                    bisectorcut.append((i, j))#connect the edge than!!!
    return bisectorcut


def cut_2_min_parts(graph):
    copy = graph.copy()
    copy = part_graph(copy)

    partition_i = get_cluster(copy, 0)
    partition_j = get_cluster(copy, 1)
    connected_copy = assign_edges(partition_i, partition_j, copy)
    return connected_copy

#find weights
def find_w(graph, edges):
    weights = []
    for edge in edges:
        weights.append(graph[edge[0]][edge[1]]['weight'])
    return weights

#find weights when graph is spearated into 2 pieces!
def find_separation_weights(graph, cluster):

    subgraph = graph.subgraph(cluster)#subgraph is our cluster
    connected_edges = cut_2_min_parts(subgraph)
    #calcualting weights now
    weights = find_w(subgraph, connected_edges)
    return weights

if __name__ == "__main__":
    # get a set of data points
    dataset = pandas.read_csv('bookexample.csv', sep=' ')
    # returns a pands.dataframe of cluster
    res = CHAMELEON_START(K=4, K_N=10, M=20, A=2.0, dataset_pandas=dataset)
    # drawing the graph with clusters
    if (len(dataset.columns) > 3):
        print("Plot Waring: more than 2-Dimensions!")
    dataset.plot(kind='scatter', c=dataset['cluster'], cmap='gist_rainbow', x=0, y=1)
    matplotlib.pyplot.show(block=False)
    matplotlib.pyplot.show()
#20 40 0.7
# 10 20 2.0
# 4 7 15 1.5