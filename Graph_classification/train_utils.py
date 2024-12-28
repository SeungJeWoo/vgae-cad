from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import os


def label_smoothing(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_labels = true_labels.type(torch.int64)
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist


def get_Adj_matrix(batch_graph):
    edge_mat_list = []
    start_idx = [0]
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))
        edge_mat_list.append(graph.edge_mat + start_idx[i])

    Adj_block_idx = np.concatenate(edge_mat_list, 1)
    # Adj_block_elem = np.ones(Adj_block_idx.shape[1])

    Adj_block_idx_row = Adj_block_idx[0,:]
    Adj_block_idx_cl = Adj_block_idx[1,:]

    return Adj_block_idx_row, Adj_block_idx_cl


def get_graphpool(batch_graph, device):
    start_idx = [0]
    # compute the padded neighbor list
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))

    idx = []
    elem = []
    for i, graph in enumerate(batch_graph):
        elem.extend([1] * len(graph.g))
        idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

    elem = torch.FloatTensor(elem)
    idx = torch.LongTensor(idx).transpose(0, 1)
    graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

    return graph_pool.to(device)


def get_batch_data(batch_graph, device):
    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    X_concat = torch.from_numpy(X_concat).to(device)
    # graph-level sum pooling

    adjj = np.concatenate([graph.edge_mat for graph in batch_graph], 0)
    adjj = torch.from_numpy(adjj).to(device)

    graph_labels = np.array([graph.label for graph in batch_graph])
    graph_labels = torch.from_numpy(graph_labels).to(device)

    return X_concat, graph_labels, adjj.to(torch.int64)


def cross_entropy(pred, soft_targets): # use nn.CrossEntropyLoss if not using soft labels in Line 159
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def train(mmodel, optimizer, train_graphs, batch_size, num_classes, device):
    # Turn on the train mode
    mmodel.train()
    indices = np.arange(0, len(train_graphs))
    np.random.shuffle(indices)
    for start in range(0, len(train_graphs), batch_size):
        end = start + batch_size
        selected_idx = indices[start:end]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        # load graph batch
        X_concat, graph_labels, adjj = get_batch_data(batch_graph, device=device)
        graph_labels = label_smoothing(graph_labels, num_classes)
        optimizer.zero_grad()
        # model probability scores
        prediction_scores = mmodel(adjj, X_concat)

        loss = cross_entropy(prediction_scores, graph_labels)
        # backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mmodel.parameters(), 0.5)  # prevent the exploding gradient problem
        optimizer.step()



def evaluate(mmodel, current_graphs, batch_size, num_classes, device, out_dir, last_round=False):
    # Turn on the evaluation mode
    mmodel.eval()
    total_loss = 0.
    with torch.no_grad():
        # evaluating
        prediction_output = []
        idx = np.arange(len(current_graphs))
        for i in range(0, len(current_graphs), batch_size):
            sampled_idx = idx[i:i + batch_size]
            if len(sampled_idx) == 0:
                continue
            batch_test_graphs = [current_graphs[j] for j in sampled_idx]
            # load graph batch
            test_X_concat, test_graph_labels, test_adj = get_batch_data(batch_test_graphs, device=device)
            # model probability scores
            prediction_scores = mmodel(test_adj, test_X_concat)

            test_graph_labels = label_smoothing(test_graph_labels, num_classes)
            loss = cross_entropy(prediction_scores, test_graph_labels)
            total_loss += loss.item()
            prediction_output.append(prediction_scores.detach())

    # model probabilities output
    prediction_output = torch.cat(prediction_output, 0)
    # predicted labels
    predictions = prediction_output.max(1, keepdim=True)[1]
    # real labels
    labels = torch.LongTensor([graph.label for graph in current_graphs]).to(device)
    # num correct predictions
    correct = predictions.eq(labels.view_as(predictions)).sum().cpu().item()
    accuracy = correct / float(len(current_graphs))

    # confusion matrix and class accuracy
    matrix = confusion_matrix(np.array(labels.cpu()), np.array(predictions.cpu()))
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    acc_x_class = matrix.diagonal() * 100

    if last_round:
        # plot and save statistics
        print("Accuracy per class :")
        print(acc_x_class)
        with open(out_dir + "/test_results.txt", 'w') as f:
            f.write("Evaluate: loss on test: "+ str(total_loss/len(current_graphs)) + " and accuracy: " + str(accuracy * 100)+"\n")
            f.write("Accuracy per class : "+ str(matrix.diagonal())+"\n")
            f.write(metrics.classification_report(np.array(labels.cpu()), np.array(predictions.cpu()), digits=3))

        ax = sns.heatmap(matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix')
        plt.savefig(out_dir + "/Confusion Matrix")

    return total_loss/len(current_graphs), accuracy, acc_x_class


def train_AE(mmodel, optimizer, train_graphs, batch_size, num_classes, device):
    # Turn on the train mode
    mmodel.train()
    indices = np.arange(0, len(train_graphs))
    np.random.shuffle(indices)
    for start in range(0, len(train_graphs), batch_size):
        end = start + batch_size
        selected_idx = indices[start:end]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        # load graph batch
        X_concat, graph_labels, adjj = get_batch_data(batch_graph, device=device)
        optimizer.zero_grad()
        # model output
        #print(adjj.shape)
        #decoded_adj, decoded_feat = mmodel(adjj, X_concat)
        decoded_feat = mmodel(adjj, X_concat)

        #loss = torch.nn.MSELoss()(decoded_adj, adjj)
        loss = torch.nn.MSELoss()(decoded_feat, X_concat)
        # backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mmodel.parameters(), 0.5)  # prevent the exploding gradient problem
        optimizer.step()

def evaluate_AE_v2(mmodel, current_graphs, batch_size, num_classes, device, out_dir, last_round=False, type_dict=None):
    # Turn on the evaluation mode
    mmodel.eval()
    total_loss = 0.
    graphCnt = 0
    if last_round:
        os.makedirs(out_dir+"/generated_graphml", exist_ok=True)
    with torch.no_grad():
        # evaluating
        prediction_output = []
        idx = np.arange(len(current_graphs))
        for i in range(0, len(current_graphs), batch_size):
            sampled_idx = idx[i:i + batch_size]
            if len(sampled_idx) == 0:
                continue
            batch_test_graphs = [current_graphs[j] for j in sampled_idx]
            # load graph batch
            test_X_concat, test_graph_labels, test_adj = get_batch_data(batch_test_graphs, device=device)
            # model MSE loss
            #decoded_adj, decoded_feat = mmodel(test_adj, test_X_concat)
            decoded_feat = mmodel(test_adj, test_X_concat)

            #loss = torch.nn.MSELoss(reduction='sum')(decoded_adj, test_adj)
            loss = torch.nn.MSELoss(reduction='sum')(decoded_feat, test_X_concat)
            total_loss += loss.item()
            if last_round:
                #save generated graphml file
                if batch_size == 1:
                    '''
                    decoded_graph = nx.Graph()

                    num_nodes = len(decoded_adj[j])
                    decoded_graph.add_nodes_from(range(num_nodes))
                    edges = [(k,l) for k in range(num_nodes) for l in range(num_nodes) if decoded_adj[k,l] > 0.5]
                    decoded_graph.add_edges_from(edges)

                    for k in range(num_nodes):
                        decoded_graph.nodes[k]['feature'] = decoded_feat[j][k]
                    '''
                    #print(test_adj.shape)
                    #print(decoded_feat.shape)
                    decoded_graph = nx.Graph()

                    num_nodes = decoded_feat.shape[0]
                    decoded_graph.add_nodes_from(range(num_nodes))
                    added_edges = set()
                    for i in range(test_adj.shape[1]):
                        node1, node2 = int(test_adj[0,i]), int(test_adj[1,i])

                        # to remove duplicated connections
                        if (node1,node2) not in added_edges and (node2,node1) not in added_edges:
                            decoded_graph.add_edge(node1,node2)
                            added_edges.add((node1,node2))
                    #edges = [(test_adj[0,k],test_adj[1,k]) for k in range(test_adj.shape[1])]
                    #decoded_graph.add_edges_from(edges)

                    #for k in range(num_nodes):
                    #    decoded_graph.nodes[k]['feature'] = decoded_feat[k].cpu().numpy().tolist()
                    #decoded_feat_dict = {k: str(decoded_feat[k].cpu().numpy().tolist()) for k in range(num_nodes)}
                    if type_dict:
                        decoded_type = {k: type_dict[np.argmax(decoded_feat[k].cpu().numpy().tolist())] for k in range(num_nodes)}
                    else:
                        decoded_type = {k: np.argmax(decoded_feat[k].cpu().numpy().tolist()) for k in range(num_nodes)}
                    nx.set_node_attributes(decoded_graph, decoded_type, 'type')

                    newfilename, _ = os.path.splitext(current_graphs[graphCnt].name_graph)
                    nx.write_graphml(decoded_graph, out_dir+"/generated_graphml/v2gen_"+newfilename)
                    print("graphml created: ", out_dir+"/generated_graphml/v2gen_"+newfilename)
                    graphCnt += 1


    #print("Total MSE loss: ", total_loss)
    #print("Total num graphs: ", len(current_graphs))


    if last_round:
        # plot to check final for visualization
        plt.subplot(1,2,1)
        nx.draw(current_graphs[graphCnt-1].g, with_labels=False, node_size=20, node_color='skyblue', font_size=4)
        plt.title("Original Graph")
        plt.subplot(1,2,2)
        nx.draw(decoded_graph, with_labels=False, node_size=20, node_color='orange', font_size=4)
        plt.title("Decoded Graph")

        plt.show()

    return total_loss/len(current_graphs)

def evaluate_AE(mmodel, current_graphs, batch_size, num_classes, device, out_dir, last_round=False):
    # Turn on the evaluation mode
    mmodel.eval()
    total_loss = 0.
    graphCnt = 0
    if last_round:
        os.makedirs(out_dir+"/generated_graphml", exist_ok=True)
    with torch.no_grad():
        # evaluating
        prediction_output = []
        idx = np.arange(len(current_graphs))
        for i in range(0, len(current_graphs), batch_size):
            sampled_idx = idx[i:i + batch_size]
            if len(sampled_idx) == 0:
                continue
            batch_test_graphs = [current_graphs[j] for j in sampled_idx]
            # load graph batch
            test_X_concat, test_graph_labels, test_adj = get_batch_data(batch_test_graphs, device=device)
            # model MSE loss
            #decoded_adj, decoded_feat = mmodel(test_adj, test_X_concat)
            decoded_feat = mmodel(test_adj, test_X_concat)

            #loss = torch.nn.MSELoss(reduction='sum')(decoded_adj, test_adj)
            loss = torch.nn.MSELoss(reduction='sum')(decoded_feat, test_X_concat)
            total_loss += loss.item()
            if last_round:
                #save generated graphml file
                if batch_size == 1:
                    '''
                    decoded_graph = nx.Graph()

                    num_nodes = len(decoded_adj[j])
                    decoded_graph.add_nodes_from(range(num_nodes))
                    edges = [(k,l) for k in range(num_nodes) for l in range(num_nodes) if decoded_adj[k,l] > 0.5]
                    decoded_graph.add_edges_from(edges)

                    for k in range(num_nodes):
                        decoded_graph.nodes[k]['feature'] = decoded_feat[j][k]
                    '''
                    #print(test_adj.shape)
                    #print(decoded_feat.shape)
                    decoded_graph = nx.Graph()

                    num_nodes = decoded_feat.shape[0]
                    decoded_graph.add_nodes_from(range(num_nodes))
                    added_edges = set()
                    for i in range(test_adj.shape[1]):
                        node1, node2 = int(test_adj[0,i]), int(test_adj[1,i])

                        # to remove duplicated connections
                        if (node1,node2) not in added_edges and (node2,node1) not in added_edges:
                            decoded_graph.add_edge(node1,node2)
                            added_edges.add((node1,node2))
                    #edges = [(test_adj[0,k],test_adj[1,k]) for k in range(test_adj.shape[1])]
                    #decoded_graph.add_edges_from(edges)

                    #for k in range(num_nodes):
                    #    decoded_graph.nodes[k]['feature'] = decoded_feat[k].cpu().numpy().tolist()
                    decoded_feat_dict = {k: str(decoded_feat[k].cpu().numpy().tolist()) for k in range(num_nodes)}
                    nx.set_node_attributes(decoded_graph, decoded_feat_dict, 'feature')

                    newfilename, _ = os.path.splitext(current_graphs[graphCnt].name_graph)
                    nx.write_graphml(decoded_graph, out_dir+"/generated_graphml/gen_"+newfilename)
                    print("graphml created: ", out_dir+"/generated_graphml/gen_"+newfilename)
                    graphCnt += 1


    #print("Total MSE loss: ", total_loss)
    #print("Total num graphs: ", len(current_graphs))


    if last_round:
        # plot to check final for visualization
        plt.subplot(1,2,1)
        nx.draw(current_graphs[graphCnt-1].g, with_labels=False, node_size=20, node_color='skyblue', font_size=4)
        plt.title("Original Graph")
        plt.subplot(1,2,2)
        nx.draw(decoded_graph, with_labels=False, node_size=20, node_color='orange', font_size=4)
        plt.title("Decoded Graph")

        plt.show()

    return total_loss/len(current_graphs)

