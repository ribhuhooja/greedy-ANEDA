# Path Approximation

**Approximate distance between any pair of nodes in the graph**

- Paper: Shortest path distance approximation using deep learning techniques (2018):
  &nbsp; [Paper](https://arxiv.org/abs/2002.05257) &nbsp;
  &nbsp;   [SourceCode](https://github.com/nayash/shortest-distance-approx-deep-learning)

    + Ideas of the paper:<br>
      1\. Read data file -> graph<br>
      2\. Use Node2Vec: nodes -> embedding<br>
      3\. Create labels<br>
      &nbsp; &nbsp; \+ 3.1 Pick a few nodes, make them as landmark nodes (randomly)<br>
      &nbsp; &nbsp; \+ 3.2 Calculate the distance for every pair (`l`, `n`),
      (`l` is a landmark node, `n` is a node in the graph)<br>
      4\. Create datasets with the labels from the previous step<br>
      5\. Make a model, try to predict the distance between 2 nodes<br>
      6\. Evaluate results<br>


- [Source Code from Sarel](https://github.com/kryptokommunist/path-length-approximation-deep-learning)
- [New Source Code (this repo)](https://github.com/BU-Lisp/dl-hyperbolic-random-graphs/tree/main/path_approximation)
    + Quick start:
      Modify the 2 config files in `configs` folder, then:
      ```
      pip install -r requirements.txt
      cd src
      python main.py
      ```
      `main.py` contains the flow step by step.
    + Current ideas:
      Similar to the 2018 paper, modify step 3 and 4:

      3'. <br>
      &nbsp; + 3'.a. Use all the nodes as landmark nodes<br>
      &nbsp; + 3'.b. Still pick some nodes as landmarks, but instead of choosing randomly,
      we [choose high-degree nodes](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/msr-tr-2009-84.pdf) <br>
      4'. Use all data as the training set (try to overfit the model)

#### TODO

- [x] Create datasets
- [x] Linear model
- [x] Neural Net
- [x] Add routing code
- [ ] More experiments
    - [ ] [Graphs](https://networkrepository.com): Different types of graphs (e.g., road networks, internet graphs),
      different sizes (small vs. large graphs)
    - [ ] Node2Vec: More dimensions (e.g., 256, 512,...), different `#epochs` to train node2vec. Any relations between
      type of graph and `#embedding_dimensions`?
    - [ ] Model: network's architecture, `#epochs`,..
    - [ ] ...
- [ ] Evaluation

#### Notes

- If you get some errors with `dgl`, try to use `conda` to re-install it:
  `conda install -c dglteam dgl`. Refer to [here](https://docs.dgl.ai/en/0.7.x/install/index.html) for more detail.
