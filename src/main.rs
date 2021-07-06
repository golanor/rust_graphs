/*
The idea of this module is to do some basic experimentation with graphs.
First, we need to define a graph and test it.
 */

mod graphs {
    use rand::prelude::*;
    use std::cmp::PartialEq;
    use std::collections::HashMap;
    use std::collections::HashSet;

    #[derive(PartialEq, Clone, Copy, Debug, Hash, Eq)]
    pub struct Node {
        id: u64,
    }

    #[derive(Hash, Eq, PartialEq, Debug)]
    pub struct Edge {
        from: Node,
        to: Node,
    }

    pub struct Graph {
        pub nodes: Vec<Node>,
        pub edges: HashSet<Edge>,
    }

    impl Graph {
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

        pub fn erdos_renyi_graph(number_of_nodes: u64, prob: f64) -> Graph {
            let node_list = (0u64..number_of_nodes)
                .map(|x| Node { id: x })
                .collect::<Vec<_>>();
            let mut edge_list: HashSet<Edge> = HashSet::new();
            for (n1, n2) in node_list
                .iter()
                .flat_map(|y| node_list.clone().into_iter().map(move |x| (x, y)))
            {
                let u: f64 = random();
                if u <= prob {
                    edge_list.insert(Edge { to: *n2, from: n1 });
                }
            }
            return Graph::from_edge_list(edge_list);
        }

        pub fn small_word_graph(
            number_of_nodes: u64,
            mut avg_node_degree: i64,
            prob: f64,
        ) -> Graph {
            let node_list = (0u64..number_of_nodes)
                .map(|x| Node { id: x })
                .collect::<Vec<_>>();
            if avg_node_degree.rem_euclid(2) != 0 {
                avg_node_degree += 1
            }
            let neighbor_indices = (-avg_node_degree / 2..avg_node_degree / 2).collect::<Vec<_>>();
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
                                number_of_nodes - y.id - (x.abs() as u64)
                            })
                            .rem_euclid(number_of_nodes),
                        },
                        y,
                    )
                })
            }) {
                let u: f64 = random();
                if u <= prob {
                    edge_list.insert(Edge { to: *n2, from: n1 });
                } else {
                    let mut random_node = Node {
                        id: rng.gen_range(0u64..number_of_nodes),
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
    }
}

#[cfg(test)]
mod tests {
    use crate::graphs::Graph;

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
