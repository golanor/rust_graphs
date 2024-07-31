/*
The idea of this module is to do some basic experimentation with graphs.
First, we need to define a graph and test it.
 */

mod graphs {
    use nalgebra::DMatrix;
    use rand::prelude::*;
    use std::cmp::PartialEq;
    use std::collections::{HashMap, HashSet};
    use std::ops::AddAssign;

    #[derive(PartialEq, Clone, Copy, Debug, Hash, Eq)]
    pub struct Node {
        id: u64,
    }

    #[derive(Hash, Eq, PartialEq, Debug)]
    pub struct Edge {
        from: Node,
        to: Node,
    }

    impl Edge {
        pub fn new_from_u64(from: u64, to: u64) -> Edge {
            Edge {
                from: Node { id: from },
                to: Node { id: to },
            }
        }

        pub fn new_from_nodes(from: Node, to: Node) -> Edge {
            Edge { from, to }
        }
    }

    /// A graph is designed here as a set of edges and a vector of nodes.
    pub struct Graph {
        pub nodes: Vec<Node>,
        pub edges: HashSet<Edge>,
    }

    impl Graph {
        /// Creates a graph from a given edge list.
        pub fn from_edge_list(edge_list: HashSet<Edge>) -> Graph {
            let mut node_list: Vec<Node> = Vec::new();
            for edge in edge_list.iter() {
                if !node_list.contains(&edge.from) {
                    node_list.push(edge.from);
                }
                if !node_list.contains(&edge.to) {
                    node_list.push(edge.to);
                }
            }
            return Graph {
                nodes: node_list,
                edges: edge_list,
            };
        }

        /// Generates aa random graph according to the Erdos-Renyi distribution.
        /// The graph is undirected, with n nodes and p probability of an edge between any two nodes.
        pub fn erdos_renyi_graph(n: u64, p: f64) -> Graph {
            let node_list = (0u64..n).map(|x| Node { id: x }).collect::<Vec<_>>();
            let mut edge_list: HashSet<Edge> = HashSet::new();
            for (n1, n2) in node_list
                .iter()
                .flat_map(|y| node_list.clone().into_iter().map(move |x| (x, y)))
            {
                let u: f64 = random();
                if u <= p {
                    edge_list.insert(Edge { to: *n2, from: n1 });
                }
            }
            return Graph::from_edge_list(edge_list);
        }

        /// Generates a small word graph, based on the paper by Watts and Strogatz.
        /// The graph is undirected, with n nodes and p as the probability of an edge between any two nodes.
        /// The average node degree is given by k.
        pub fn small_word_graph(n: u64, mut k: i64, p: f64) -> Graph {
            let node_list = (0u64..n).map(|x| Node { id: x }).collect::<Vec<_>>();
            if k.rem_euclid(2) != 0 {
                k += 1
            }
            let neighbor_indices = (-k / 2..k / 2).collect::<Vec<_>>();
            let mut edge_list: HashSet<Edge> = HashSet::new();
            let mut rng = rand::thread_rng();
            for (n1, n2) in node_list.iter().flat_map(|y| {
                neighbor_indices.clone().into_iter().map(move |x| {
                    (
                        Node {
                            id: (if x > 0 {
                                (x as u64) + y.id
                            } else if (x.abs() as u64) < y.id {
                                y.id - (x.abs() as u64)
                            } else {
                                n - y.id - (x.abs() as u64)
                            })
                            .rem_euclid(n),
                        },
                        y,
                    )
                })
            }) {
                let u: f64 = random();
                if u <= p {
                    edge_list.insert(Edge { to: *n2, from: n1 });
                } else {
                    let mut random_node = Node {
                        id: rng.gen_range(0u64..n),
                    };
                    if random_node.id == n1.id {
                        random_node.id += 1
                    }
                    edge_list.insert(Edge {
                        from: n1,
                        to: random_node,
                    });
                }
            }
            return Graph::from_edge_list(edge_list);
        }

        /// Generate a complete graph with n nodes.
        pub fn complete_graph(n: u64) -> Graph {
            let node_list = (0u64..n).map(|x| Node { id: x }).collect::<Vec<_>>();
            let edge_list: HashSet<Edge> = node_list
                .iter()
                .flat_map(|y| {
                    node_list
                        .clone()
                        .into_iter()
                        .map(move |x| Edge::new_from_nodes(*y, x))
                })
                .collect();
            return Graph::from_edge_list(edge_list);
        }

        /// Returns the adjacency matrix of the graph.
        pub fn adjacency_matrix<
            T: 'static
                + std::marker::Copy
                + std::fmt::Debug
                + PartialEq
                + num_traits::One
                + num_traits::Zero
                + AddAssign,
        >(
            &self,
        ) -> DMatrix<T> {
            let number_of_nodes = self.nodes.len() as u64;
            let vector_size = self.nodes.len() * self.nodes.len();
            let mut adj_list = vec![T::zero(); vector_size];
            for edge in self.edges.iter() {
                adj_list[(edge.from.id * number_of_nodes + edge.to.id) as usize] += T::one();
            }
            let adjacency_matrix =
                DMatrix::<T>::from_vec(self.nodes.len(), self.nodes.len(), adj_list);
            // let adjacency_matrix = (adjacency_matrix + adjacency_matrix.transpose()) / (T::one() + T::one());
            return adjacency_matrix;
        }

        /// Calculates the average node degree of the graph.
        pub fn average_node_degree(&self) -> f64 {
            let mut counter: HashMap<Node, u64> = HashMap::new();
            for edge in self.edges.iter() {
                let val = counter.entry(edge.from).or_insert(0);
                *val += 1;
                let val2 = counter.entry(edge.to).or_insert(0);
                *val2 += 1;
            }
            return (counter.values().sum::<u64>() as f64) / (counter.keys().len() as f64);
        }

        /// Calculates the Katz index, which is a measure of the centrality of a node.
        /// It calculates the number of paths between two nodes, u and v.
        /// We add a discount factor, beta, which reduces the weight of longer paths.
        pub fn katz_index_matrix(&self, beta: f64) -> DMatrix<f64> {
            let adjacency_matrix = self.adjacency_matrix();
            let identity_matrix =
                DMatrix::<f64>::identity(adjacency_matrix.nrows(), adjacency_matrix.ncols());
            let katz_matrix = (identity_matrix.clone() - beta * adjacency_matrix)
                .try_inverse()
                .unwrap()
                - identity_matrix;
            katz_matrix
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::graphs::{Edge, Graph};
    use nalgebra::matrix;
    use std::collections::HashSet;

    #[test]
    fn number_of_nodes_er_is_correct() {
        let n = 30;
        let er_graph = Graph::erdos_renyi_graph(n, 0.7);
        assert_eq!(n as usize, er_graph.nodes.len());
    }
    #[test]
    fn number_of_nodes_sw_is_correct() {
        let n = 30;
        let graph = Graph::small_word_graph(n, 2, 0.7);
        assert_eq!(n as usize, graph.nodes.len());
    }
    #[test]
    fn graph_from_edge_list_is_correct() {
        let edge_list: HashSet<Edge> = vec![
            Edge::new_from_u64(1, 2),
            Edge::new_from_u64(0, 2),
            Edge::new_from_u64(1, 3),
            Edge::new_from_u64(2, 3),
        ]
        .into_iter()
        .collect();
        let graph = Graph::from_edge_list(edge_list);
        assert_eq!(graph.edges.len(), 4);
    }
    #[test]
    fn graph_adjacency_matrix_is_correct() {
        let edge_list: HashSet<Edge> = vec![
            Edge::new_from_u64(1, 2),
            Edge::new_from_u64(0, 2),
            Edge::new_from_u64(1, 3),
            Edge::new_from_u64(2, 3),
        ]
        .into_iter()
        .collect();
        let graph = Graph::from_edge_list(edge_list);
        let adj_matrix = graph.adjacency_matrix::<u32>();
        let correct_adj_matrix = matrix![0, 0, 1, 0; 0, 0, 1, 1; 0, 0, 0, 1; 0, 0, 0, 0];
    }
}

fn main() {
    use crate::graphs::Graph;
    let new_graph = Graph::erdos_renyi_graph(16, 0.5);
    println!("{}", new_graph.edges.len());
    let new_new_graph = Graph::small_word_graph(16, 2, 0.3);
    println!("{}", new_new_graph.edges.len());
    println!(
        "Average graph degree {}",
        new_new_graph.average_node_degree()
    )
}
