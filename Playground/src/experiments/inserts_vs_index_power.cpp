/**
 * inserts_vs_index_power
 *
 * This experiment studies how the structure of the model changes as more
 * and more inserts are received. Specifically, we are interested in looking
 * at the average size of a leaf node as a measure of how powerful the structure
 * is. We run this experiment for a variety of configurations with a specified
 * initial size and number of inserts. We also compare the indexing power to a
 * naive model which is trained on the initial data + inserts to see how our
 * model degrades compared to the baseline.
 *
 * Input:
 * - Epsilon
 * - Epsilon recursive
 * - n = initial data size
 * - num_inserts = total number of inserts to perform
 * - granularity = how many inserts to perform at a time up to num_inserts
 * - List of configurations, where each configuration specifies:
 *    - name = column name in output
 *    - fill_ratio
 *    - fill_ratio_recursive
 *    - max_buffer_size
 *    - split_neighborhood
 * - seeds = list of seeds to run this experiment on
 *
 * Output:
 * - name = name of the configuration generating this row, or "baseline"
 * - data_size
 */

// Input
// < Define all the Inputs listed above>

void inserts_vs_index_power()
{
  return;
}