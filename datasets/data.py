from torch_geometric import data

# -------------------------# -------------------------# -------------------------
from custom_data_ops import Connect
# -------------------------# -------------------------# -------------------------


class Data(data.Data):
    def __init__(self,
                 x=None,
                 edge_index=None,
                 edge_attr=None,
                 y=None,
                 v_outs=None,
                 e_outs=None,
                 g_outs=None,
                 o_outs=None,
                 laplacians=None,
                 v_plus=None,
                 edge_index_mod=None,  # 31/1 TODO: modification of FA layer
                 **kwargs):
        additional_fields = {
            'v_outs': v_outs,
            'e_outs': e_outs,
            'g_outs': g_outs,
            'o_outs': o_outs,
            'laplacians': laplacians,
            'v_plus': v_plus,
            'edge_index_mod': edge_index_mod  # 31/1 TODO: modification of FA layer
        }

        super().__init__(x, edge_index, edge_attr, y, **additional_fields)


class Batch(data.Batch):

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        laplacians = None
        v_plus = None

        if 'laplacians' in data_list[0]:
            laplacians = [d.laplacians[:] for d in data_list]
            v_plus = [d.v_plus[:] for d in data_list]

        copy_data = []
        # --------------- IDO -----------
        copy_data_mod = []  # 31/1 TODO: WILL HOLD MODIFIED DATA WITH AUG. FA LAYER
        for d in data_list:
            # 18/1 TODO: CHANGED THIS SINCE CAUSED BUG, SOME FIELDS DON'T EXIST (DEPRECATED)
            # copy_data.append(Data(x=d.x,
            #                       y=d.y,
            #                       edge_index=d.edge_index,
            #                       edge_attr=d.edge_attr,
            #                       v_outs=d.v_outs,
            #                       g_outs=d.g_outs,
            #                       e_outs=d.e_outs,
            #                       o_outs=d.o_outs)
            #                  )
            # ---------------------- IDO ---------------------------
            # 31/1 TODO: the stuff we changed with CC and FA modifications
            # coo = sparse.coo_matrix((np.ones_like(d.edge_index.numpy()[0, :]),
            #                          (d.edge_index.numpy()[0, :], d.edge_index[1, :])),
            #                         shape=(d.x.numpy().shape[0], d.x.numpy().shape[0]))
            # graph = csr_matrix(coo.toarray())
            # n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

            connect = Connect(d)
            d_new = connect.reduced_cliques_complement() # TODO!!!: CASE NO EDGES AFTER AUG. USE FA LAYER OR REGULAR ADJ. MAT?
            edge_index_mod = d_new.edge_index
            copy_data_mod.append(Data(x=d.x,
                                      y=d.y,
                                      edge_index=edge_index_mod,
                                      )
                                 )

            # ---------------------- IDO ---------------------------
            copy_data.append(Data(x=d.x,
                                  y=d.y,
                                  edge_index=d.edge_index,
                                  )
                             )

        batch = data.Batch.from_data_list(copy_data, follow_batch=follow_batch)
        batch['laplacians'] = laplacians
        batch['v_plus'] = v_plus
        # ---------------- 31/1 TODO: create mod batch for proper transfer of edges to batch graph -----------------
        batch_mod = data.Batch.from_data_list(copy_data_mod, follow_batch=follow_batch)
        # ----------------------------------------------------------------------------------------------
        batch['edge_index_mod'] = batch_mod.edge_index

        return batch
