from spark_env.env import Environment
import numpy as np
from spark_env.job_dag import JobDAG
from spark_env.node import Node
from msg_passing_path import *
import bisect
from tf_op import leaky_relu
from gcn import GraphCNN
import sys
from gsn import GraphSNN
import tensorflow as tf
from matplotlib import pyplot as plt
import networkx as nx

env = Environment()
tf.compat.v1.disable_eager_execution()

reset_prob = 5e-7
max_time = np.random.geometric(reset_prob)
env.seed(234)
env.reset(max_time=max_time)
obs = env.observe()
node_input_dim = 5
job_input_dim = 3
hid_dims = [16, 8]
output_dim = 8
max_depth = 8
exec_cap  = 50
eps=1e-6
executor_levels = range(1, exec_cap + 1)
act_fn=leaky_relu
postman = Postman()

def translate_state(obs):
        """
        Translate the observation to matrix form
        """
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map = obs

        # compute total number of nodes
        total_num_nodes = int(np.sum([job_dag.num_nodes for job_dag in job_dags]))

        # job and node inputs to feed
        node_inputs = np.zeros([total_num_nodes, node_input_dim])
        job_inputs = np.zeros([len(job_dags), job_input_dim])

        # sort out the exec_map
        exec_map = {}
        for job_dag in job_dags:
            exec_map[job_dag] = len(job_dag.executors)
        # count in moving executors
        for node in moving_executors.moving_executors.values():
            exec_map[node.job_dag] += 1
        # count in executor commit
        for s in exec_commit.commit:
            if isinstance(s, JobDAG):
                j = s
            elif isinstance(s, Node):
                j = s.job_dag
            elif s is None:
                j = None
            else:
                print('source', s, 'unknown')
                exit(1)
            for n in exec_commit.commit[s]:
                if n is not None and n.job_dag != j:
                    exec_map[n.job_dag] += exec_commit.commit[s][n]

        # gather job level inputs
        job_idx = 0
        for job_dag in job_dags:
            # number of executors in the job
            job_inputs[job_idx, 0] = exec_map[job_dag] / 20.0
            # the current executor belongs to this job or not
            if job_dag is source_job:
                job_inputs[job_idx, 1] = 2
            else:
                job_inputs[job_idx, 1] = -2
            # number of source executors
            job_inputs[job_idx, 2] = num_source_exec / 20.0

            job_idx += 1

        # gather node level inputs
        node_idx = 0
        job_idx = 0
        for job_dag in job_dags:
            for node in job_dag.nodes:

                # copy the feature from job_input first
                node_inputs[node_idx, :3] = job_inputs[job_idx, :3]

                # work on the node
                node_inputs[node_idx, 3] = \
                    (node.num_tasks - node.next_task_idx) * \
                    node.tasks[-1].duration / 100000.0

                # number of tasks left
                node_inputs[node_idx, 4] = \
                    (node.num_tasks - node.next_task_idx) / 200.0

                node_idx += 1

            job_idx += 1

        return node_inputs, job_inputs, \
               job_dags, source_job, num_source_exec, \
               frontier_nodes, executor_limits, \
               exec_commit, moving_executors, \
               exec_map, action_map

def get_valid_masks(job_dags, frontier_nodes,
            source_job, num_source_exec, exec_map, action_map):

        job_valid_mask = np.zeros([1, \
            len(job_dags) * len(executor_levels)])

        job_valid = {}  # if job is saturated, don't assign node

        base = 0
        for job_dag in job_dags:
            # new executor level depends on the source of executor
            if job_dag is source_job:
                least_exec_amount = \
                    exec_map[job_dag] - num_source_exec + 1
                    # +1 because we want at least one executor
                    # for this job
            else:
                least_exec_amount = exec_map[job_dag] + 1
                # +1 because of the same reason above

            assert least_exec_amount > 0
            assert least_exec_amount <= executor_levels[-1] + 1

            # find the index for first valid executor limit
            exec_level_idx = bisect.bisect_left(
                executor_levels, least_exec_amount)

            if exec_level_idx >= len(executor_levels):
                job_valid[job_dag] = False
            else:
                job_valid[job_dag] = True

            for l in range(exec_level_idx, len(executor_levels)):
                job_valid_mask[0, base + l] = 1

            base += executor_levels[-1]

        total_num_nodes = int(np.sum([job_dag.num_nodes for job_dag in job_dags]))

        node_valid_mask = np.zeros([1, total_num_nodes])

        for node in frontier_nodes:
            if job_valid[node.job_dag]:
                act = action_map.inverse_map[node]
                node_valid_mask[0, act] = 1

        return node_valid_mask, job_valid_mask

node_inputs, job_inputs, job_dags, source_job, \
    num_source_exec, frontier_nodes, \
    executor_limits, exec_commit, \
    moving_executors, exec_map, action_map = translate_state(obs)


for job in job_dags:
    nx.draw(job.get_networkx(), with_labels = True)
    plt.show()
    break

# print(node_inputs) # array of 5 floats
# print(job_inputs) # array of 3 floats
# print(source_job) # None


""" gcn_mats, gcn_masks, dag_summ_backward_map,running_dags_mat, job_dags_changed = postman.get_msg_path(job_dags)
node_valid_mask, job_valid_mask = get_valid_masks(job_dags, frontier_nodes,source_job, num_source_exec, exec_map, action_map) """






# batch_size = tf.shape(input=node_valid_mask)[0]
# node_inputs_reshape = tf.reshape(node_inputs, [batch_size, -1, node_input_dim])
# job_inputs_reshape = tf.reshape(job_inputs, [batch_size, -1, job_input_dim])
# gcn_outputs_reshape = tf.reshape(gcn.outputs, [batch_size, -1, output_dim])
# gsn_dag_summ_reshape = tf.reshape(gsn.summaries[0], [batch_size, -1, output_dim])
# gsn_summ_backward_map_extend = tf.tile(tf.expand_dims(gsn_summ_backward_map, axis=0), [batch_size, 1, 1])
# gsn_dag_summ_extend = tf.matmul(gsn_summ_backward_map_extend, gsn_dag_summ_reshape)
# gsn_global_summ_reshape = tf.reshape(gsn.summaries[1], [batch_size, -1, output_dim])
# gsn_global_summ_extend_job = tf.tile(gsn_global_summ_reshape, [1, tf.shape(input=gsn_dag_summ_reshape)[1], 1])
# gsn_global_summ_extend_node = tf.tile(gsn_global_summ_reshape, [1, tf.shape(input=gsn_dag_summ_extend)[1], 1])


# merge_node = tf.concat([
#                 node_inputs_reshape, gcn_outputs_reshape,
#                 gsn_dag_summ_extend,
#                 gsn_global_summ_extend_node], axis=2)


# print(merge_nodes)

# for job in job_dags:
#     print(job)
# print("\n\n")

# print(source_job, "\n\n")

# print(num_source_exec, "\n\n")

# for node in frontier_nodes:
#     print(node)
# print("\n\n")

# print(executor_limits, "\n\n")

# print(exec_commit, "\n\n")

# print(moving_executors.moving_executors)
# print(moving_executors.node_track, "\n\n")

# print(action_map.map)
# print(action_map.inverse_map, "\n\n")